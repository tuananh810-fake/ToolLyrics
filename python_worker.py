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


def _smooth_line_word_timings(
    words: list[dict[str, Any]],
    *,
    line_start_ms: int,
    line_end_limit_ms: int,
    min_duration_ms: int = MIN_EXPORT_WORD_DURATION_MS,
) -> list[dict[str, Any]]:
    """Make exported karaoke timings renderable without changing token text.

    stable-ts can occasionally return zero-length or near-zero word spans. The
    JSON still looks precise, but a word-by-word renderer cannot highlight a
    0ms token. This pass keeps timings monotonic, preserves strong anchors as
    much as possible, and redistributes only the spans that are too short.
    """
    if not words:
        return []

    normalized: list[dict[str, Any]] = []
    for index, word in enumerate(words):
        start_ms = _coerce_ms(word.get("s"), line_start_ms)
        end_ms = _coerce_ms(word.get("e"), start_ms)
        if index == 0:
            start_ms = max(line_start_ms, start_ms)
        normalized.append({**word, "s": start_ms, "e": max(start_ms, end_ms)})

    for index, word in enumerate(normalized):
        previous_end = normalized[index - 1]["e"] if index else line_start_ms
        start_ms = max(_coerce_ms(word.get("s"), previous_end), previous_end)

        if index + 1 < len(normalized):
            next_start = _coerce_ms(normalized[index + 1].get("s"), start_ms + min_duration_ms)
        else:
            next_start = max(line_end_limit_ms, start_ms + min_duration_ms)

        desired_end = max(_coerce_ms(word.get("e"), start_ms), start_ms + min_duration_ms)
        hard_limit = max(start_ms, next_start)
        if desired_end > hard_limit:
            desired_end = hard_limit

        word["s"] = start_ms
        word["e"] = max(start_ms, desired_end)

    # Second pass: if crowded anchors still produced tiny spans, distribute the
    # whole line evenly inside the available line window. This is preferable to
    # exporting 0ms words because it keeps the renderer stable and readable.
    if any(_coerce_ms(word.get("e")) - _coerce_ms(word.get("s")) < min_duration_ms for word in normalized):
        first_start = max(line_start_ms, _coerce_ms(normalized[0].get("s"), line_start_ms))
        last_end = max(line_end_limit_ms, _coerce_ms(normalized[-1].get("e"), first_start))
        total_available = max(last_end - first_start, len(normalized) * min_duration_ms)
        step = total_available / len(normalized)
        cursor = first_start
        for index, word in enumerate(normalized):
            start_ms = int(round(cursor))
            if index == len(normalized) - 1:
                end_ms = max(start_ms + min_duration_ms, int(round(first_start + total_available)))
            else:
                end_ms = max(start_ms + min_duration_ms, int(round(first_start + step * (index + 1))))
            word["s"] = start_ms
            word["e"] = end_ms
            cursor = end_ms

    return normalized


def postprocess_lyrics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    lines = payload.get("data")
    if not isinstance(lines, list):
        return payload

    for index, line in enumerate(lines):
        if not isinstance(line, dict):
            continue
        words = line.get("words")
        if not isinstance(words, list) or not words:
            continue

        line_start_ms = _coerce_ms(line.get("startTimeMs"), _coerce_ms(words[0].get("s"), 0))
        fallback_end_ms = max(
            _coerce_ms(words[-1].get("e"), line_start_ms),
            line_start_ms + len(words) * MIN_EXPORT_WORD_DURATION_MS,
        )
        line_end_limit_ms = _next_line_start_ms(lines, index, fallback_end_ms)
        if line_end_limit_ms <= line_start_ms:
            line_end_limit_ms = fallback_end_ms

        smoothed_words = _smooth_line_word_timings(
            words,
            line_start_ms=line_start_ms,
            line_end_limit_ms=line_end_limit_ms,
        )
        line["words"] = smoothed_words
        if smoothed_words:
            line["startTimeMs"] = smoothed_words[0]["s"]

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
