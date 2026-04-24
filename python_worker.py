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


def send_message(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


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
