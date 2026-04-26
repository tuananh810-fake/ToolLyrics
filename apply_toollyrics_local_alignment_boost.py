#!/usr/bin/env python3
"""
Local alignment-success patch for tuananh810-fake/ToolLyrics.
Run this script from the repository root:

    python apply_toollyrics_local_alignment_boost.py

It backs up edited files as *.bak.local-boost before writing changes.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from textwrap import dedent

PATCH_MARKER = "TOOLLYRICS_LOCAL_ALIGNMENT_BOOST_V1"


def read_text(path: Path) -> str:
    data = path.read_bytes()
    text = data.decode("utf-8-sig", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def write_text(path: Path, text: str) -> None:
    backup = path.with_suffix(path.suffix + ".bak.local-boost")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    path.write_text(text, encoding="utf-8", newline="\n")


def replace_function(text: str, name: str, next_name: str, replacement: str, file_label: str) -> str:
    pattern = rf"(?ms)^def {re.escape(name)}\(.*?(?=^def {re.escape(next_name)}\()"
    new_text, count = re.subn(pattern, replacement.rstrip() + "\n\n", text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not replace function {name}() in {file_label}.")
    return new_text


def patch_arg_default(text: str, flag: str, new_default: str) -> str:
    pattern = rf'(?ms)(parser\.add_argument\(\s*"{re.escape(flag)}".*?\bdefault=)([^,\n]+)(,)'
    new_text, count = re.subn(pattern, rf"\g<1>{new_default}\g<3>", text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not update default for {flag}.")
    return new_text


NORMALIZE_BLOCK = r'''
# TOOLLYRICS_LOCAL_ALIGNMENT_BOOST_V1: extra lyric cleanup and tolerant tokenization.
LOCAL_BOOST_INLINE_TIME_TAG_RE = re.compile(r"<\s*\d+(?::\d+){0,2}(?:[\.,]\d+)?\s*>")
LOCAL_BOOST_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
LOCAL_BOOST_SINGER_PREFIX_RE = re.compile(r"^\s*[\w .\-\(\)\[\]]{1,32}\s*[:\uff1a]\s+(?=\S)", re.UNICODE)
LOCAL_BOOST_DECORATION_RE = re.compile(r"^[\s\-\u2010-\u2015_=.*~|/\\]+|[\s\-\u2010-\u2015_=.*~|/\\]+$")
LOCAL_BOOST_MOJIBAKE_MARKERS = ("\u00c3", "\u00c2", "\u00e2\u20ac", "\u00c6", "\u00c4", "\u00c5")
LOCAL_BOOST_MARKER_WORDS = {
    "instrumental", "inst", "intro", "interlude", "solo", "outro", "music", "beat",
    "nhac", "dao nhac", "nhac dao", "hoa tau", "melody",
}


def local_boost_mojibake_score(value: str) -> int:
    return sum(value.count(marker) for marker in LOCAL_BOOST_MOJIBAKE_MARKERS)


def repair_mojibake_text(text: str) -> str:
    value = str(text or "").replace("\ufeff", "").replace("\u00a0", " ")
    value = value.replace("\u00e2\u20ac\u2122", "'").replace("\u00e2\u20ac\u02dc", "'")
    value = value.replace("\u00e2\u20ac\u0153", '"').replace("\u00e2\u20ac\u009d", '"')
    value = value.replace("\u00e2\u20ac\u201c", "-").replace("\u00e2\u20ac\u201d", "-")
    if any(marker in value for marker in LOCAL_BOOST_MOJIBAKE_MARKERS):
        original_score = local_boost_mojibake_score(value)
        for encoding in ("latin1", "cp1252"):
            try:
                candidate = value.encode(encoding, errors="strict").decode("utf-8", errors="strict")
            except (UnicodeEncodeError, UnicodeDecodeError, LookupError):
                continue
            if candidate and local_boost_mojibake_score(candidate) < original_score:
                value = candidate
                original_score = local_boost_mojibake_score(value)
    return value


def clean_lyric_text(text: str) -> str:
    value = repair_mojibake_text(text)
    value = unicodedata.normalize("NFKC", value)
    value = LOCAL_BOOST_ZERO_WIDTH_RE.sub("", value)
    value = value.replace("\u3000", " ")
    value = LOCAL_BOOST_INLINE_TIME_TAG_RE.sub(" ", value)
    value = value.replace("\u266a", " ").replace("\u266b", " ").replace("\u266c", " ")
    value = value.replace("/", " / ").replace("|", " ")
    value = LOCAL_BOOST_DECORATION_RE.sub("", value)
    value = collapse_spaces(value)
    prefix = LOCAL_BOOST_SINGER_PREFIX_RE.match(value)
    if prefix:
        label = prefix.group(0).strip(" :\uff1a")
        if 0 < len(label.split()) <= 4 and len(value[prefix.end():].split()) >= 2:
            value = value[prefix.end():]
    return collapse_spaces(value)


def normalize_token(token: str) -> str:
    cleaned = clean_lyric_text(token).lower()
    cleaned = cleaned.replace("\u2019", "'").replace("\u2018", "'").replace("`", "'").replace("\u00b4", "'")
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
    cleaned = re.sub(r"[\u2010-\u2015]", "-", cleaned)
    cleaned = EDGE_PUNCT_RE.sub("", cleaned)
    cleaned = cleaned.strip("-_.,!?;:\"'()[]{}<>")
    return cleaned
'''


TOKENIZE_BLOCK = r'''
def tokenize_lyrics(text: str) -> list[str]:
    collapsed = clean_lyric_text(text)
    if not collapsed:
        return []
    collapsed = re.sub(r"\s*/\s*", " ", collapsed)
    return [token for token in collapsed.split(" ") if normalize_token(token)]
'''


PARSE_LRC_BLOCK = r'''
def read_lyric_text_fallback(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1258", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def parse_lrc_offset_ms(metadata: dict[str, str]) -> int:
    raw_offset = str(metadata.get("offset", "0") or "0").strip().replace(",", ".")
    try:
        return int(round(float(raw_offset)))
    except ValueError:
        print(f"[warn] Invalid LRC offset value {raw_offset!r}; using 0 ms.", file=sys.stderr)
        return 0


def is_instrumental_line(text: str) -> bool:
    collapsed = collapse_spaces(clean_lyric_text(text))
    if not collapsed:
        return True
    folded = strip_accents(collapsed.lower())
    words = folded.split()
    if len(words) <= 6 and any(marker in folded for marker in LOCAL_BOOST_MARKER_WORDS):
        return True
    if len(words) <= 3 and folded.strip("()[]{} .-_") in LOCAL_BOOST_MARKER_WORDS:
        return True
    return False


def parse_lrc(lrc_path: Path) -> tuple[dict[str, str], list[LrcLine]]:
    metadata: dict[str, str] = {}
    parsed_lines: list[tuple[float, str, str]] = []
    seen: set[tuple[int, str]] = set()

    for raw_line in read_lyric_text_fallback(lrc_path).splitlines():
        line = repair_mojibake_text(raw_line).strip()
        if not line:
            continue
        meta_match = META_TAG_RE.fullmatch(line)
        if meta_match and not TIME_TAG_RE.search(line):
            metadata[meta_match.group(1).lower()] = repair_mojibake_text(meta_match.group(2)).strip()
            continue
        tags = list(TIME_TAG_RE.finditer(line))
        if not tags:
            continue
        lyric_text = clean_lyric_text(TIME_TAG_RE.sub(" ", line))
        for tag in tags:
            timestamp_s = parse_lrc_timestamp(tag)
            key = (round(timestamp_s * 1000), lyric_text)
            if key in seen:
                continue
            seen.add(key)
            parsed_lines.append((timestamp_s, lyric_text, raw_line))

    if not parsed_lines:
        raise ValueError(f"No timestamped lyric lines found in {lrc_path}")

    offset_s = parse_lrc_offset_ms(metadata) / 1000.0
    parsed_lines.sort(key=lambda item: item[0])
    lrc_lines = [
        LrcLine(
            index=index,
            timestamp_s=max(0.0, timestamp_s + offset_s),
            text=lyric_text,
            raw_text=raw_text,
            is_instrumental=is_instrumental_line(lyric_text),
        )
        for index, (timestamp_s, lyric_text, raw_text) in enumerate(parsed_lines)
    ]
    return metadata, lrc_lines
'''


BUILD_SEGMENTS_BLOCK = r'''
def build_alignment_segments(
    lines: list[LrcLine],
    windows: list[LineWindow],
    matches: list[LineMatch],
    segment_pad_s: float,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    segments: list[dict[str, Any]] = []
    line_to_segment: dict[int, int] = {}
    extra_pad_s = min(max(segment_pad_s * 2.0, 0.35), 1.25)

    for line, window, match in zip(lines, windows, matches, strict=True):
        tokens = tokenize_lyrics(line.text)
        if line.is_instrumental or not tokens:
            continue

        base_start_s = max(0.0, window.start_s - extra_pad_s)
        base_end_s = max(window.end_s + extra_pad_s, window.start_s + 0.25)

        if match.has_anchor:
            segment_start_s = max(base_start_s, float(match.anchor_start_s or window.start_s) - segment_pad_s)
            segment_end_s = min(base_end_s, float(match.anchor_end_s or window.end_s) + segment_pad_s)
        else:
            segment_start_s = base_start_s
            segment_end_s = base_end_s

        if segment_end_s <= segment_start_s:
            segment_end_s = max(base_end_s, segment_start_s + 0.25)

        line_to_segment[line.index] = len(segments)
        segments.append({"start": round(segment_start_s, 3), "end": round(segment_end_s, 3), "text": clean_lyric_text(line.text)})

    return segments, line_to_segment
'''


RUN_FORCED_BLOCK = r'''
def run_forced_alignment(
    model: Any,
    audio_path: str,
    segments: list[dict[str, Any]],
    language: str | None,
    min_word_dur: float,
    refine_precision: float,
    skip_refine: bool,
    *,
    stage_label: str,
    use_vad: bool,
    progress_callback: ProgressCallback | None = None,
    progress_stage: str = "forced_alignment",
    progress_start: int = 0,
    progress_end: int = 100,
) -> Any | None:
    if not segments:
        return None

    last_reported_percent = -1

    def stable_progress(done_s: float, total_s: float) -> None:
        nonlocal last_reported_percent
        if total_s <= 0:
            return
        ratio = min(max(done_s / total_s, 0.0), 1.0)
        percent = progress_start + round((progress_end - progress_start) * ratio)
        if percent <= last_reported_percent:
            return
        last_reported_percent = percent
        report_progress(progress_callback, progress_stage, f"Running {stage_label}.", percent, details={"doneSeconds": round(done_s, 2), "totalSeconds": round(total_s, 2), "segmentCount": len(segments)})

    attempts: list[tuple[bool, float, str]] = [(use_vad, min_word_dur, "primary")]
    if use_vad:
        attempts.append((False, min_word_dur, "retry_without_vad"))
    if min_word_dur > 0.015:
        attempts.append((False, max(0.01, min_word_dur / 2.0), "retry_short_words"))

    last_error: Exception | None = None
    for attempt_index, (attempt_vad, attempt_min_word_dur, attempt_name) in enumerate(attempts, start=1):
        try:
            report_progress(progress_callback, progress_stage, f"Running {stage_label} ({attempt_name}).", max(progress_start, min(progress_end, progress_start + attempt_index - 1)), details={"segmentCount": len(segments), "attempt": attempt_name, "vad": attempt_vad, "minWordDur": round(attempt_min_word_dur, 3)})
            aligned_result = align_segments_to_audio(model, audio_path, segments, language, attempt_min_word_dur, use_vad=attempt_vad, progress_callback=stable_progress if progress_callback else None)
            if not skip_refine:
                aligned_result = safe_refine_result(model, audio_path, aligned_result, refine_precision, stage_label=stage_label)
            return aligned_result
        except Exception as exc:
            last_error = exc
            print(f"[warn] {stage_label} attempt {attempt_name} failed; trying next fallback if available: {exc}", file=sys.stderr)

    print(f"[warn] {stage_label} failed after {len(attempts)} attempt(s): {last_error}", file=sys.stderr)
    return None
'''


SAFE_SAVE_BLOCK = r'''
def safe_save_stable_json(result: Any, path: Path, label: str) -> None:
    if result is None or not hasattr(result, "save_as_json"):
        return
    try:
        import contextlib
        with contextlib.redirect_stdout(sys.stderr):
            result.save_as_json(str(path))
    except UnicodeEncodeError as exc:
        print(f"[warn] Skipping {label} debug JSON because stable-ts could not encode text: {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"[warn] Skipping {label} debug JSON: {exc}", file=sys.stderr)
'''


SEND_MESSAGE_BLOCK = r'''
def send_message(payload: dict[str, Any]) -> None:
    output = getattr(sys, "__stdout__", None) or sys.stdout
    output.write(json.dumps(payload, ensure_ascii=False) + "\n")
    output.flush()
'''


READ_PLAIN_BLOCK = r'''
def _read_plain_lyric_lines(path: str) -> list[str]:
    raw = Path(path).read_bytes()
    text = None
    for encoding in ("utf-8-sig", "utf-8", "cp1258", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if text is None:
        text = raw.decode("utf-8", errors="replace")

    lines = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.strip().split())
        if line:
            lines.append(line)
    if not lines:
        raise ValueError("TXT lyric file is empty. Add one lyric line per row.")
    return lines
'''


HANDLE_ALIGN_BLOCK = r'''
def handle_align(message_id: str, payload: dict[str, Any]) -> None:
    requested_device = str(payload.get("device", "auto"))
    requested_model = str(payload.get("model", "recommended"))
    resolved_device = resolve_device_choice(requested_device)
    resolved_model = resolve_model_choice(requested_model, resolved_device)

    def progress_callback(stage: str, message: str, percent: int, details: dict[str, Any] | None) -> None:
        send_message({"id": message_id, "event": "progress", "progress": {"stage": stage, "message": message, "percent": percent, "details": details or {}}})

    import contextlib
    with contextlib.redirect_stdout(sys.stderr):
        if str(payload.get("lyricsMode", "lrc")) == "plain_text":
            result_payload = run_plain_text_alignment(payload, resolved_model=resolved_model, resolved_device=resolved_device, progress_callback=progress_callback)
        else:
            result_payload = run_alignment(
                payload["audioPath"], payload["lrcPath"], payload["outputPath"],
                song=payload.get("song"), model_name=resolved_model, device=resolved_device,
                language=payload.get("language"), window_pad_ms=int(payload.get("windowPadMs", 650)),
                segment_pad_ms=int(payload.get("segmentPadMs", 350)), tail_pad_ms=int(payload.get("tailPadMs", 8000)),
                min_word_dur=float(payload.get("minWordDur", 0.03)), refine_precision=float(payload.get("refinePrecision", 0.04)),
                skip_refine=bool(payload.get("skipRefine", False)), strategy=str(payload.get("strategy", "auto")),
                use_vad=bool(payload.get("useVad", True)), debug_dir=payload.get("debugDir"), progress_callback=progress_callback,
            )
        result_payload = postprocess_lyrics_payload(result_payload)
        write_processed_payload(payload["outputPath"], result_payload)

    send_message({"id": message_id, "ok": True, "payload": result_payload, "resolvedDevice": resolved_device, "resolvedModel": resolved_model, "outputPath": str(Path(payload["outputPath"]).resolve())})
'''


def patch_high_precision(path: Path) -> bool:
    text = read_text(path)
    if PATCH_MARKER in text:
        print(f"[skip] {path} already contains {PATCH_MARKER}")
        return False

    text = text.replace("from __future__ import annotations\n", f"from __future__ import annotations\n# {PATCH_MARKER}\n", 1)
    text = patch_arg_default(text, "--window-pad-ms", "650")
    text = patch_arg_default(text, "--segment-pad-ms", "350")
    text = patch_arg_default(text, "--tail-pad-ms", "8000")
    text = replace_function(text, "normalize_token", "folded_token", NORMALIZE_BLOCK, str(path))
    text = replace_function(text, "tokenize_lyrics", "parse_lrc_timestamp", TOKENIZE_BLOCK, str(path))
    text = replace_function(text, "is_instrumental_line", "parse_lrc", "", str(path))
    text = replace_function(text, "parse_lrc", "token_similarity", PARSE_LRC_BLOCK, str(path))
    text = replace_function(text, "build_alignment_segments", "collect_expected_token_count", BUILD_SEGMENTS_BLOCK, str(path))
    text = text.replace("similarity < 0.72", "similarity < 0.66")
    text = replace_function(text, "run_forced_alignment", "write_debug_outputs", RUN_FORCED_BLOCK, str(path))
    text = replace_function(text, "safe_save_stable_json", "build_output_payload", SAFE_SAVE_BLOCK, str(path))
    write_text(path, text)
    return True


def patch_worker(path: Path) -> bool:
    text = read_text(path)
    if PATCH_MARKER in text:
        print(f"[skip] {path} already contains {PATCH_MARKER}")
        return False

    text = text.replace("from __future__ import annotations\n", f"from __future__ import annotations\n# {PATCH_MARKER}\n", 1)
    text = text.replace("MIN_TXT_MATCH_SIMILARITY = 0.72", "MIN_TXT_MATCH_SIMILARITY = 0.66")
    text = replace_function(text, "send_message", "_coerce_ms", SEND_MESSAGE_BLOCK, str(path))
    text = replace_function(text, "_read_plain_lyric_lines", "_fill_plain_text_line_timings", READ_PLAIN_BLOCK, str(path))
    text = replace_function(text, "handle_align", "main", HANDLE_ALIGN_BLOCK, str(path))
    write_text(path, text)
    return True


def patch_server(path: Path) -> bool:
    text = read_text(path)
    if PATCH_MARKER in text:
        print(f"[skip] {path} already contains {PATCH_MARKER}")
        return False

    text = text.replace("const path = require(\"path\");\n", f"const path = require(\"path\");\n// {PATCH_MARKER}\n", 1)
    text = re.sub(
        r'const strategy = lyricsMode === "plain_text" \? "transcribe" : \(body\.qualityFallback === "on" \? "auto" : "direct"\);',
        'const strategy = lyricsMode === "plain_text" ? "transcribe" : (body.qualityFallback === "off" ? "direct" : "auto");',
        text,
        count=1,
    )
    text = text.replace(
        "handleWorkerMessage(JSON.parse(line));",
        """const trimmed = line.trim();\n      if (!trimmed.startsWith(\"{\")) {\n        console.error(\"[python-worker]\", trimmed);\n        continue;\n      }\n      handleWorkerMessage(JSON.parse(trimmed));""",
        1,
    )
    text = re.sub(
        r'debugDir: job\.debugEnabled \? job\.debugPath : undefined\s*}',
        dedent('''\
        debugDir: job.debugEnabled ? job.debugPath : undefined,
        windowPadMs: 650,
        segmentPadMs: 350,
        tailPadMs: 8000,
        minWordDur: 0.03,
        refinePrecision: job.skipRefine ? 0.04 : 0.03
      }''').rstrip(),
        text,
        count=1,
    )
    write_text(path, text)
    return True


def main() -> int:
    root = Path.cwd()
    targets = (
        ("aligner", patch_high_precision, root / "high_precision_word_aligner.py"),
        ("worker", patch_worker, root / "python_worker.py"),
        ("server", patch_server, root / "server.js"),
    )
    missing = [str(path) for _, _, path in targets if not path.exists()]
    if missing:
        print("Run this script from the ToolLyrics repository root. Missing: " + ", ".join(missing), file=sys.stderr)
        return 2

    changed = []
    for label, func, path in targets:
        try:
            if func(path):
                changed.append(label)
                print(f"[ok] patched {path}")
        except Exception as exc:
            print(f"[error] failed to patch {path}: {exc}", file=sys.stderr)
            return 1

    if changed:
        print("[done] Patched: " + ", ".join(changed))
        print("[done] Backups were written as *.bak.local-boost")
        print("[next] Restart the Node server and run one LRC job with strategy auto.")
    else:
        print("[done] No changes needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
