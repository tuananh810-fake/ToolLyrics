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
    get_alignment_model,
    get_runtime_profile,
    resolve_device_choice,
    resolve_model_choice,
    run_alignment,
)


MIN_EXPORT_WORD_DURATION_MS = 80
MAX_REASONABLE_WORD_DURATION_MS = 2200


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
    """Repair only unsafe word spans while preserving model anchors.

    The previous export guard made every word renderable, but it could flatten a
    whole line when just a few stable-ts spans were too short. This version is
    more conservative: keep valid model timings, redistribute only consecutive
    bad spans, then do a final monotonic repair pass. Full-line redistribution
    is used only when almost every token in the line is unusable.
    """
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

    # Remove overlaps without destroying gaps. Gaps are useful silence/vocal
    # boundaries, so only push a word forward when it starts before the previous
    # word has ended.
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

    # If the whole line is unreliable, distribute the line evenly once. This is
    # a last resort for empty/poor word-level timestamps, not the normal path.
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

    # Final monotonic repair. If a cluster expanded into the next model anchor,
    # push only the downstream overlap forward; do not flatten the full line.
    cursor = line_start_ms
    for index, word in enumerate(normalized):
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
