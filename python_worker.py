from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from high_precision_word_aligner import (
    extract_words_from_result,
    get_alignment_model,
    get_runtime_profile,
    resolve_device_choice,
    resolve_model_choice,
    run_alignment,
    safe_refine_result,
    semiglobal_align,
    tokenize_lyrics,
    transcribe_audio,
)


MIN_EXPORT_WORD_DURATION_MS = 80
MAX_REASONABLE_WORD_DURATION_MS = 2200
DEFAULT_TXT_WORD_DURATION_MS = 360
MIN_TXT_MATCH_SIMILARITY = 0.72


def send_message(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _coerce_ms(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(round(float(value))))
    except (TypeError, ValueError):
        return default


def _next_line_start_ms(lines: list[dict[str, Any]], index: int, fallback_ms: int) -> int:
    for next_line in lines[index + 1 :]:
        words = next_line.get("words")
        if isinstance(words, list) and words:
            return _coerce_ms(words[0].get("s"), fallback_ms)
        if "startTimeMs" in next_line:
            return _coerce_ms(next_line.get("startTimeMs"), fallback_ms)
    return fallback_ms


def _redistribute_range(
    words: list[dict[str, Any]],
    start_index: int,
    end_index: int,
    left_bound_ms: int,
    right_bound_ms: int,
    *,
    min_duration_ms: int,
) -> None:
    count = end_index - start_index + 1
    if count <= 0:
        return

    right_bound_ms = max(right_bound_ms, left_bound_ms + count * min_duration_ms)
    total_duration = right_bound_ms - left_bound_ms
    step = total_duration / count
    cursor = left_bound_ms

    for offset, word_index in enumerate(range(start_index, end_index + 1)):
        start_ms = int(round(cursor))
        if offset == count - 1:
            end_ms = right_bound_ms
        else:
            end_ms = int(round(left_bound_ms + step * (offset + 1)))
        end_ms = max(start_ms + min_duration_ms, end_ms)
        words[word_index]["s"] = start_ms
        words[word_index]["e"] = end_ms
        cursor = end_ms


def _smooth_line_word_timings(
    words: list[dict[str, Any]],
    *,
    line_start_ms: int,
    line_end_limit_ms: int,
    min_duration_ms: int = MIN_EXPORT_WORD_DURATION_MS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Repair only unsafe word spans while preserving model anchors."""
    stats = {
        "adjustedWords": 0,
        "redistributedClusters": 0,
        "fullLineRedistributed": False,
        "quality": "model",
    }
    if not words:
        return [], stats

    normalized: list[dict[str, Any]] = []
    for index, word in enumerate(words):
        start_ms = _coerce_ms(word.get("s"), line_start_ms)
        end_ms = _coerce_ms(word.get("e"), start_ms)
        if index == 0:
            start_ms = max(line_start_ms, start_ms)
        normalized.append({**word, "s": start_ms, "e": max(start_ms, end_ms)})

    cursor = line_start_ms
    for word in normalized:
        original_start = word["s"]
        original_end = word["e"]
        word["s"] = max(word["s"], cursor)
        word["e"] = max(word["s"], word["e"])
        if word["s"] != original_start or word["e"] != original_end:
            stats["adjustedWords"] += 1
        cursor = word["e"]

    bad_indices = [
        index
        for index, word in enumerate(normalized)
        if word["e"] - word["s"] < min_duration_ms
    ]

    if len(bad_indices) >= max(2, int(len(normalized) * 0.75)):
        first_start = max(line_start_ms, normalized[0]["s"])
        last_end = max(line_end_limit_ms, normalized[-1]["e"], first_start + len(normalized) * min_duration_ms)
        _redistribute_range(
            normalized,
            0,
            len(normalized) - 1,
            first_start,
            last_end,
            min_duration_ms=min_duration_ms,
        )
        stats["adjustedWords"] += len(normalized)
        stats["redistributedClusters"] += 1
        stats["fullLineRedistributed"] = True
        stats["quality"] = "estimated"
        return normalized, stats

    index = 0
    while index < len(normalized):
        duration_ms = normalized[index]["e"] - normalized[index]["s"]
        if duration_ms >= min_duration_ms:
            index += 1
            continue

        cluster_start = index
        while index + 1 < len(normalized) and normalized[index + 1]["e"] - normalized[index + 1]["s"] < min_duration_ms:
            index += 1
        cluster_end = index

        left_bound = normalized[cluster_start - 1]["e"] if cluster_start > 0 else line_start_ms
        right_bound = normalized[cluster_end + 1]["s"] if cluster_end + 1 < len(normalized) else line_end_limit_ms
        if right_bound <= left_bound:
            right_bound = left_bound + (cluster_end - cluster_start + 1) * min_duration_ms

        _redistribute_range(
            normalized,
            cluster_start,
            cluster_end,
            left_bound,
            right_bound,
            min_duration_ms=min_duration_ms,
        )
        stats["adjustedWords"] += cluster_end - cluster_start + 1
        stats["redistributedClusters"] += 1
        if stats["quality"] == "model":
            stats["quality"] = "smoothed"
        index += 1

    cursor = line_start_ms
    for word in normalized:
        original_start = word["s"]
        original_end = word["e"]
        word["s"] = max(word["s"], cursor)
        word["e"] = max(word["s"] + min_duration_ms, word["e"])
        if word["e"] - word["s"] > MAX_REASONABLE_WORD_DURATION_MS:
            word["e"] = word["s"] + MAX_REASONABLE_WORD_DURATION_MS
        if word["s"] != original_start or word["e"] != original_end:
            stats["adjustedWords"] += 1
            if stats["quality"] == "model":
                stats["quality"] = "smoothed"
        cursor = word["e"]

    return normalized, stats


def postprocess_lyrics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    lines = payload.get("data")
    if not isinstance(lines, list):
        return payload

    total_adjusted_words = 0
    total_smoothed_lines = 0
    total_estimated_lines = 0

    for index, line in enumerate(lines):
        if not isinstance(line, dict):
            continue
        words = line.get("words")
        if not isinstance(words, list) or not words:
            line["timingQuality"] = "empty"
            continue

        line_start_ms = _coerce_ms(line.get("startTimeMs"), _coerce_ms(words[0].get("s"), 0))
        fallback_end_ms = max(
            _coerce_ms(words[-1].get("e"), line_start_ms),
            line_start_ms + len(words) * MIN_EXPORT_WORD_DURATION_MS,
        )
        line_end_limit_ms = _next_line_start_ms(lines, index, fallback_end_ms)
        if line_end_limit_ms <= line_start_ms:
            line_end_limit_ms = fallback_end_ms

        smoothed_words, stats = _smooth_line_word_timings(
            words,
            line_start_ms=line_start_ms,
            line_end_limit_ms=line_end_limit_ms,
        )
        line["words"] = smoothed_words
        line["timingQuality"] = stats["quality"]
        line["timingFix"] = {
            "adjustedWords": stats["adjustedWords"],
            "redistributedClusters": stats["redistributedClusters"],
            "fullLineRedistributed": stats["fullLineRedistributed"],
            "minWordDurationMs": MIN_EXPORT_WORD_DURATION_MS,
        }
        total_adjusted_words += int(stats["adjustedWords"])
        if stats["quality"] == "smoothed":
            total_smoothed_lines += 1
        if stats["quality"] == "estimated":
            total_estimated_lines += 1
        if smoothed_words:
            line["startTimeMs"] = smoothed_words[0]["s"]

    payload["timingSummary"] = {
        "minWordDurationMs": MIN_EXPORT_WORD_DURATION_MS,
        "adjustedWords": total_adjusted_words,
        "smoothedLines": total_smoothed_lines,
        "estimatedLines": total_estimated_lines,
    }
    return payload


def _read_plain_lyric_lines(path: str) -> list[str]:
    text = Path(path).read_text(encoding="utf-8-sig")
    lines = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.strip().split())
        if line:
            lines.append(line)
    if not lines:
        raise ValueError("TXT lyric file is empty. Add one lyric line per row.")
    return lines


def _fill_plain_text_line_timings(
    tokens: list[str],
    token_timings: list[tuple[int | None, int | None]],
    cursor_ms: int,
) -> list[dict[str, int | str]]:
    anchors = [index for index, (start, end) in enumerate(token_timings) if start is not None and end is not None]
    if not anchors:
        words = []
        for index, token in enumerate(tokens):
            start_ms = cursor_ms + index * DEFAULT_TXT_WORD_DURATION_MS
            words.append({"w": token, "s": start_ms, "e": start_ms + DEFAULT_TXT_WORD_DURATION_MS})
        return words

    starts = [start for start, _ in token_timings]
    ends = [end for _, end in token_timings]
    first_anchor = anchors[0]
    last_anchor = anchors[-1]

    if first_anchor > 0:
        right = int(starts[first_anchor] or cursor_ms)
        left = max(cursor_ms, right - first_anchor * DEFAULT_TXT_WORD_DURATION_MS)
        step = (right - left) / first_anchor
        for index in range(first_anchor):
            starts[index] = int(round(left + step * index))
            ends[index] = int(round(left + step * (index + 1)))

    for left_anchor, right_anchor in zip(anchors, anchors[1:]):
        gap_count = right_anchor - left_anchor - 1
        if gap_count <= 0:
            continue
        left = int(ends[left_anchor] or starts[left_anchor] or cursor_ms)
        right = int(starts[right_anchor] or left)
        if right <= left:
            right = left + gap_count * DEFAULT_TXT_WORD_DURATION_MS
        step = (right - left) / gap_count
        for offset, index in enumerate(range(left_anchor + 1, right_anchor)):
            starts[index] = int(round(left + step * offset))
            ends[index] = int(round(left + step * (offset + 1)))

    if last_anchor < len(tokens) - 1:
        left = int(ends[last_anchor] or starts[last_anchor] or cursor_ms)
        for offset, index in enumerate(range(last_anchor + 1, len(tokens)), start=0):
            starts[index] = left + offset * DEFAULT_TXT_WORD_DURATION_MS
            ends[index] = left + (offset + 1) * DEFAULT_TXT_WORD_DURATION_MS

    words = []
    running_cursor = max(cursor_ms, int(starts[0] or cursor_ms))
    for token, start_ms, end_ms in zip(tokens, starts, ends, strict=True):
        start_ms = max(running_cursor, int(start_ms if start_ms is not None else running_cursor))
        end_ms = max(start_ms + MIN_EXPORT_WORD_DURATION_MS, int(end_ms if end_ms is not None else start_ms + DEFAULT_TXT_WORD_DURATION_MS))
        words.append({"w": token, "s": start_ms, "e": end_ms})
        running_cursor = end_ms
    return words


def run_plain_text_alignment(
    payload: dict[str, Any],
    *,
    resolved_model: str,
    resolved_device: str,
    progress_callback,
) -> dict[str, Any]:
    lyric_lines = _read_plain_lyric_lines(payload["lrcPath"])
    all_reference_tokens: list[str] = []
    token_ranges: list[tuple[int, int, str]] = []
    for line in lyric_lines:
        start_index = len(all_reference_tokens)
        tokens = tokenize_lyrics(line)
        all_reference_tokens.extend(tokens)
        token_ranges.append((start_index, len(all_reference_tokens), line))

    if not all_reference_tokens:
        raise ValueError("TXT lyric file has no usable lyric words.")

    progress_callback("transcribing", "Transcribing MP3 for plain TXT lyric alignment...", 12, {"mode": "plain_text"})
    model = get_alignment_model(resolved_model, resolved_device)

    def stable_progress(done_s: float, total_s: float) -> None:
        if total_s <= 0:
            return
        percent = 12 + int(min(1.0, max(0.0, done_s / total_s)) * 45)
        progress_callback("transcribing", "Detecting vocal word timings from audio...", percent, {"mode": "plain_text"})

    result = transcribe_audio(
        model,
        payload["audioPath"],
        payload.get("language"),
        float(payload.get("minWordDur", 0.02)),
        use_vad=bool(payload.get("useVad", True)),
        progress_callback=stable_progress,
    )

    if not bool(payload.get("skipRefine", False)):
        progress_callback("refining", "Refining detected word timings...", 62, {"mode": "plain_text"})
        result = safe_refine_result(
            model,
            payload["audioPath"],
            result,
            float(payload.get("refinePrecision", 0.02)),
            stage_label="plain text transcription",
        )

    detected_words = extract_words_from_result(result)
    if not detected_words:
        raise ValueError("No word-level timestamps were detected from the audio.")

    progress_callback("aligning_text", "Aligning TXT lyrics to detected audio words...", 74, {"mode": "plain_text"})
    hypothesis_tokens = [word.text for word in detected_words]
    _, alignment = semiglobal_align(all_reference_tokens, hypothesis_tokens)

    token_to_detected_index: dict[int, int] = {}
    matched_tokens = 0
    for ref_index, hyp_index, similarity in alignment:
        if ref_index is None or hyp_index is None:
            continue
        if similarity < MIN_TXT_MATCH_SIMILARITY:
            continue
        token_to_detected_index[ref_index] = hyp_index
        matched_tokens += 1

    data: list[dict[str, Any]] = []
    cursor_ms = max(0, int(round(detected_words[0].start_s * 1000)))
    for start_index, end_index, line in token_ranges:
        tokens = all_reference_tokens[start_index:end_index]
        timings: list[tuple[int | None, int | None]] = []
        matched_in_line = 0
        for absolute_index in range(start_index, end_index):
            detected_index = token_to_detected_index.get(absolute_index)
            if detected_index is None:
                timings.append((None, None))
                continue
            detected_word = detected_words[detected_index]
            timings.append((
                int(round(detected_word.start_s * 1000)),
                int(round(detected_word.end_s * 1000)),
            ))
            matched_in_line += 1

        words = _fill_plain_text_line_timings(tokens, timings, cursor_ms)
        if words:
            cursor_ms = words[-1]["e"] + 120
        data.append(
            {
                "line": line,
                "startTimeMs": words[0]["s"] if words else cursor_ms,
                "words": words,
                "plainTextMatch": {
                    "matchedWords": matched_in_line,
                    "totalWords": len(tokens),
                    "matchRatio": round(matched_in_line / len(tokens), 3) if tokens else 1.0,
                },
            }
        )

    payload_out: dict[str, Any] = {
        "song": payload.get("song") or Path(payload["audioPath"]).stem,
        "inputMode": "plain_text",
        "plainTextSummary": {
            "lyricLines": len(lyric_lines),
            "lyricWords": len(all_reference_tokens),
            "detectedWords": len(detected_words),
            "matchedWords": matched_tokens,
            "matchRatio": round(matched_tokens / len(all_reference_tokens), 3),
        },
        "data": data,
    }
    progress_callback("postprocess", "Repairing export timings...", 90, {"mode": "plain_text"})
    return postprocess_lyrics_payload(payload_out)


def write_processed_payload(output_path: str, payload: dict[str, Any]) -> None:
    Path(output_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def handle_runtime_info(message_id: str) -> None:
    send_message(
        {
            "id": message_id,
            "ok": True,
            "runtime": get_runtime_profile(),
        }
    )


def handle_warmup(message_id: str, payload: dict[str, Any]) -> None:
    requested_device = str(payload.get("device", "auto"))
    requested_model = str(payload.get("model", "recommended"))
    resolved_device = resolve_device_choice(requested_device)
    resolved_model = resolve_model_choice(requested_model, resolved_device)
    get_alignment_model(resolved_model, resolved_device)
    send_message(
        {
            "id": message_id,
            "ok": True,
            "resolvedDevice": resolved_device,
            "resolvedModel": resolved_model,
        }
    )


def handle_align(message_id: str, payload: dict[str, Any]) -> None:
    requested_device = str(payload.get("device", "auto"))
    requested_model = str(payload.get("model", "recommended"))
    resolved_device = resolve_device_choice(requested_device)
    resolved_model = resolve_model_choice(requested_model, resolved_device)

    def progress_callback(stage: str, message: str, percent: int, details: dict[str, Any] | None) -> None:
        send_message(
            {
                "id": message_id,
                "event": "progress",
                "progress": {
                    "stage": stage,
                    "message": message,
                    "percent": percent,
                    "details": details or {},
                },
            }
        )

    if str(payload.get("lyricsMode", "lrc")) == "plain_text":
        result_payload = run_plain_text_alignment(
            payload,
            resolved_model=resolved_model,
            resolved_device=resolved_device,
            progress_callback=progress_callback,
        )
    else:
        result_payload = run_alignment(
            payload["audioPath"],
            payload["lrcPath"],
            payload["outputPath"],
            song=payload.get("song"),
            model_name=resolved_model,
            device=resolved_device,
            language=payload.get("language"),
            window_pad_ms=int(payload.get("windowPadMs", 300)),
            segment_pad_ms=int(payload.get("segmentPadMs", 180)),
            tail_pad_ms=int(payload.get("tailPadMs", 4000)),
            min_word_dur=float(payload.get("minWordDur", 0.02)),
            refine_precision=float(payload.get("refinePrecision", 0.02)),
            skip_refine=bool(payload.get("skipRefine", False)),
            strategy=str(payload.get("strategy", "auto")),
            use_vad=bool(payload.get("useVad", True)),
            debug_dir=payload.get("debugDir"),
            progress_callback=progress_callback,
        )
        result_payload = postprocess_lyrics_payload(result_payload)

    write_processed_payload(payload["outputPath"], result_payload)

    send_message(
        {
            "id": message_id,
            "ok": True,
            "payload": result_payload,
            "resolvedDevice": resolved_device,
            "resolvedModel": resolved_model,
            "outputPath": str(Path(payload["outputPath"]).resolve()),
        }
    )


def main() -> int:
    send_message(
        {
            "event": "ready",
            "runtime": get_runtime_profile(),
        }
    )

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        message_id = "unknown"
        try:
            message = json.loads(line)
            message_id = str(message.get("id", "unknown"))
            command = message.get("command")
            payload = message.get("payload", {}) or {}

            if command == "runtime_info":
                handle_runtime_info(message_id)
            elif command == "warmup":
                handle_warmup(message_id, payload)
            elif command == "align":
                handle_align(message_id, payload)
            else:
                raise ValueError(f"Unsupported worker command: {command}")
        except Exception as exc:  # pragma: no cover - IPC safety
            send_message(
                {
                    "id": message_id,
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
