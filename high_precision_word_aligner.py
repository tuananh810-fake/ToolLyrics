from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable


TIME_TAG_RE = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")
META_TAG_RE = re.compile(r"^\[([A-Za-z]+):(.*)\]\s*$")
WHITESPACE_RE = re.compile(r"\s+")
EDGE_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)
INSTRUMENTAL_MARKERS = {
    "instrumental",
    "inst",
    "intro",
    "interlude",
    "solo",
    "music",
    "nhac dao",
    "dao nhac",
    "melody",
    "beat",
}

ProgressCallback = Callable[[str, str, int, dict[str, Any] | None], None]


@dataclass(slots=True)
class TimedWord:
    text: str
    start_s: float
    end_s: float
    probability: float | None = None


@dataclass(slots=True)
class LrcLine:
    index: int
    timestamp_s: float
    text: str
    raw_text: str
    is_instrumental: bool = False


@dataclass(slots=True)
class LineWindow:
    start_s: float
    end_s: float


@dataclass(slots=True)
class LineMatch:
    matched_token_map: dict[int, int] = field(default_factory=dict)
    raw_score: float = 0.0
    confidence: float = 0.0
    anchor_start_s: float | None = None
    anchor_end_s: float | None = None

    @property
    def has_anchor(self) -> bool:
        return self.anchor_start_s is not None and self.anchor_end_s is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "High-precision word aligner for MP3 + LRC using stable-ts. "
            "Outputs lyrics_pro.json for LyricsEngine.js."
        )
    )
    parser.add_argument("--audio", required=True, help="Path to the input .mp3 audio file.")
    parser.add_argument("--lrc", required=True, help="Path to the source .lrc file.")
    parser.add_argument(
        "--output",
        default="lyrics_pro.json",
        help="Path to the output JSON file. Defaults to lyrics_pro.json.",
    )
    parser.add_argument(
        "--song",
        default=None,
        help="Override the song title in the output JSON. Defaults to [ti:] from LRC or audio stem.",
    )
    parser.add_argument(
        "--model",
        default="recommended",
        choices=("recommended", "large-v3", "medium"),
        help="Whisper model to use. Defaults to recommended (medium on CPU, large-v3 on CUDA).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device passed to stable-ts load_model(), e.g. auto, cpu, cuda.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code/name for Whisper. Example: vi, en. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--window-pad-ms",
        type=int,
        default=300,
        help="Padding added around each LRC line window when collecting ASR candidates.",
    )
    parser.add_argument(
        "--segment-pad-ms",
        type=int,
        default=180,
        help="Padding added around the anchored segment before align_words().",
    )
    parser.add_argument(
        "--tail-pad-ms",
        type=int,
        default=4000,
        help="Fallback tail padding for the last line if audio end cannot be inferred.",
    )
    parser.add_argument(
        "--min-word-dur",
        type=float,
        default=0.02,
        help="Minimum word duration passed to stable-ts. Defaults to 0.02 seconds.",
    )
    parser.add_argument(
        "--refine-precision",
        type=float,
        default=0.02,
        help=(
            "stable-ts refine precision in seconds. Values below 0.02 are clamped "
            "because stable-ts cannot refine below 20ms."
        ),
    )
    parser.add_argument(
        "--skip-refine",
        action="store_true",
        help="Skip stable-ts refine() passes.",
    )
    parser.add_argument(
        "--strategy",
        default="auto",
        choices=("auto", "direct", "transcribe"),
        help="Alignment strategy. auto prefers direct LRC forced alignment and falls back to transcription when needed.",
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Optional directory for raw transcription and alignment debug JSON.",
    )
    return parser.parse_args()


def load_stable_whisper() -> Any:
    try:
        import stable_whisper
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise SystemExit(
            "stable-ts is not installed. Install with: python -m pip install stable-ts\n"
            "Also ensure torch + ffmpeg are available for your Python runtime."
        ) from exc
    return stable_whisper


@lru_cache(maxsize=4)
def get_alignment_model(model_name: str, device: str) -> Any:
    stable_whisper = load_stable_whisper()
    ensure_ffmpeg_on_path()

    model_kwargs: dict[str, Any] = {}
    if device != "auto":
        model_kwargs["device"] = device
    if device == "cpu":
        model_kwargs["dq"] = True
    return stable_whisper.load_model(model_name, **model_kwargs)


@lru_cache(maxsize=1)
def detect_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def resolve_device_choice(device: str) -> str:
    normalized = (device or "auto").lower()
    if normalized in ("gpu", "nvidia"):
        normalized = "cuda"
    if normalized == "cuda":
        if not detect_cuda_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot see an NVIDIA GPU.")
        return "cuda"
    if normalized == "cpu":
        return "cpu"
    if normalized != "auto":
        raise ValueError(f"Unsupported device: {device}")
    return "cuda" if detect_cuda_available() else "cpu"


def resolve_model_choice(model_name: str, device: str) -> str:
    if model_name != "recommended":
        return model_name
    return "medium"


def get_runtime_profile() -> dict[str, Any]:
    resolved_device = resolve_device_choice("auto")
    torch_version = None
    cuda_version = None
    cuda_device_name = None
    try:
        import torch
        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cuda_device_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return {
        "cudaAvailable": detect_cuda_available(),
        "resolvedDevice": resolved_device,
        "recommendedModel": resolve_model_choice("recommended", resolved_device),
        "torchVersion": torch_version,
        "cudaVersion": cuda_version,
        "cudaDeviceName": cuda_device_name,
    }


def ensure_ffmpeg_on_path() -> None:
    if shutil.which("ffmpeg"):
        return
    try:
        import imageio_ffmpeg
    except ImportError:
        return

    ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    shim_dir = Path(__file__).resolve().parent / ".runtime"
    shim_dir.mkdir(parents=True, exist_ok=True)
    runtime_ffmpeg = shim_dir / "ffmpeg.exe"
    if not runtime_ffmpeg.exists() or runtime_ffmpeg.stat().st_size != ffmpeg_exe.stat().st_size:
        shutil.copy2(ffmpeg_exe, runtime_ffmpeg)

    current_path = os.environ.get("PATH", "")
    os.environ["PATH"] = str(shim_dir) + os.pathsep + current_path


def as_dict_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def collapse_spaces(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def strip_accents(text: str) -> str:
    text = text.replace("Đ", "D").replace("đ", "d")
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_token(token: str) -> str:
    cleaned = unicodedata.normalize("NFKC", token).lower()
    cleaned = cleaned.replace("’", "'").replace("‘", "'")
    cleaned = cleaned.replace("â€™", "'").replace("â€˜", "'")
    cleaned = EDGE_PUNCT_RE.sub("", cleaned)
    return cleaned


def folded_token(token: str) -> str:
    return strip_accents(normalize_token(token))


def tokenize_lyrics(text: str) -> list[str]:
    collapsed = collapse_spaces(text)
    if not collapsed:
        return []
    return collapsed.split(" ")


def parse_lrc_timestamp(match: re.Match[str]) -> float:
    minutes = int(match.group(1))
    seconds = float(match.group(2))
    return minutes * 60 + seconds


def is_instrumental_line(text: str) -> bool:
    collapsed = collapse_spaces(text)
    if not collapsed:
        return True
    folded = strip_accents(collapsed.lower())
    if len(folded.split()) <= 4 and any(marker in folded for marker in INSTRUMENTAL_MARKERS):
        return True
    if (
        len(folded.split()) <= 6
        and (folded.startswith("(") or folded.startswith("[") or folded.startswith("{"))
        and any(marker in folded for marker in INSTRUMENTAL_MARKERS)
    ):
        return True
    return False


def parse_lrc(lrc_path: Path) -> tuple[dict[str, str], list[LrcLine]]:
    metadata: dict[str, str] = {}
    parsed_lines: list[tuple[float, str, str]] = []

    for raw_line in lrc_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        meta_match = META_TAG_RE.fullmatch(line)
        if meta_match and not TIME_TAG_RE.search(line):
            metadata[meta_match.group(1).lower()] = meta_match.group(2).strip()
            continue

        tags = list(TIME_TAG_RE.finditer(line))
        if not tags:
            continue

        lyric_text = collapse_spaces(TIME_TAG_RE.sub("", line))
        for tag in tags:
            parsed_lines.append((parse_lrc_timestamp(tag), lyric_text, raw_line))

    if not parsed_lines:
        raise ValueError(f"No timestamped lyric lines found in {lrc_path}")

    offset_ms = int(metadata.get("offset", "0") or "0")
    offset_s = offset_ms / 1000.0

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


def token_similarity(left: str, right: str) -> float:
    left_norm = normalize_token(left)
    right_norm = normalize_token(right)
    if not left_norm or not right_norm:
        return 1.0 if left_norm == right_norm else 0.0
    if left_norm == right_norm:
        return 1.0

    left_fold = strip_accents(left_norm)
    right_fold = strip_accents(right_norm)
    if left_fold == right_fold:
        return 0.97

    direct_ratio = SequenceMatcher(a=left_norm, b=right_norm).ratio()
    folded_ratio = SequenceMatcher(a=left_fold, b=right_fold).ratio()
    return max(direct_ratio, folded_ratio)


def token_match_score(left: str, right: str) -> float:
    similarity = token_similarity(left, right)
    if similarity >= 0.999:
        return 4.0
    if similarity >= 0.96:
        return 3.25
    if similarity >= 0.88:
        return 2.0
    if similarity >= 0.72:
        return 0.75
    if similarity >= 0.55:
        return -0.65
    return -2.75


def semiglobal_align(reference_tokens: list[str], hypothesis_tokens: list[str]) -> tuple[float, list[tuple[int | None, int | None, float]]]:
    if not reference_tokens:
        return 0.0, []

    gap_in_reference = -1.15
    gap_in_hypothesis = -1.65
    rows = len(reference_tokens) + 1
    cols = len(hypothesis_tokens) + 1

    scores = [[-math.inf] * cols for _ in range(rows)]
    backtrack: list[list[tuple[int, int, str] | None]] = [[None] * cols for _ in range(rows)]
    scores[0][0] = 0.0

    for col in range(1, cols):
        scores[0][col] = 0.0
        backtrack[0][col] = (0, col - 1, "I")

    for row in range(1, rows):
        scores[row][0] = scores[row - 1][0] + gap_in_hypothesis
        backtrack[row][0] = (row - 1, 0, "D")

    for row in range(1, rows):
        for col in range(1, cols):
            diagonal = scores[row - 1][col - 1] + token_match_score(
                reference_tokens[row - 1], hypothesis_tokens[col - 1]
            )
            delete = scores[row - 1][col] + gap_in_hypothesis
            insert = scores[row][col - 1] + gap_in_reference

            best_score = diagonal
            best_step = (row - 1, col - 1, "M")
            if delete > best_score:
                best_score = delete
                best_step = (row - 1, col, "D")
            if insert > best_score:
                best_score = insert
                best_step = (row, col - 1, "I")

            scores[row][col] = best_score
            backtrack[row][col] = best_step

    best_col = max(range(cols), key=lambda col: scores[-1][col])
    alignment: list[tuple[int | None, int | None, float]] = []

    row = rows - 1
    col = best_col
    while row > 0 or col > 0:
        step = backtrack[row][col]
        if step is None:
            break
        prev_row, prev_col, opcode = step
        if opcode == "M":
            similarity = token_similarity(reference_tokens[row - 1], hypothesis_tokens[col - 1])
            alignment.append((row - 1, col - 1, similarity))
        elif opcode == "D":
            alignment.append((row - 1, None, 0.0))
        else:
            alignment.append((None, col - 1, 0.0))
        row, col = prev_row, prev_col

    alignment.reverse()
    return scores[-1][best_col], alignment


def extract_words_from_result(result: Any) -> list[TimedWord]:
    words: list[TimedWord] = []
    for segment in as_dict_value(result, "segments", []) or []:
        segment_words = as_dict_value(segment, "words", []) or []
        for raw_word in segment_words:
            text = collapse_spaces(str(as_dict_value(raw_word, "word", as_dict_value(raw_word, "text", ""))))
            start_s = float(as_dict_value(raw_word, "start", as_dict_value(segment, "start", 0.0)) or 0.0)
            end_s = float(as_dict_value(raw_word, "end", start_s) or start_s)
            if not normalize_token(text):
                continue
            probability = as_dict_value(raw_word, "probability", None)
            words.append(TimedWord(text=text, start_s=start_s, end_s=max(start_s, end_s), probability=probability))
    return words


def extract_segment_ranges_from_result(result: Any) -> list[tuple[float, float, str]]:
    segments: list[tuple[float, float, str]] = []
    for raw_segment in as_dict_value(result, "segments", []) or []:
        text = collapse_spaces(str(as_dict_value(raw_segment, "text", "")))
        start_s = float(as_dict_value(raw_segment, "start", 0.0) or 0.0)
        end_s = float(as_dict_value(raw_segment, "end", start_s) or start_s)
        if not text and end_s <= start_s:
            continue
        segments.append((start_s, max(start_s, end_s), text))
    return segments


def infer_result_end_s(
    result: Any,
    words: list[TimedWord],
    segment_ranges: list[tuple[float, float, str]],
    fallback_end_s: float,
) -> float:
    if words:
        return words[-1].end_s
    if segment_ranges:
        return max(end_s for _, end_s, _ in segment_ranges)
    duration = as_dict_value(result, "duration", None)
    if duration is not None:
        try:
            return max(float(duration), fallback_end_s)
        except (TypeError, ValueError):
            pass
    return fallback_end_s


def build_line_windows(lines: list[LrcLine], inferred_audio_end_s: float, tail_pad_s: float) -> list[LineWindow]:
    windows: list[LineWindow] = []
    for index, line in enumerate(lines):
        if index + 1 < len(lines):
            next_start_s = lines[index + 1].timestamp_s
        else:
            next_start_s = max(inferred_audio_end_s, line.timestamp_s + tail_pad_s)
        windows.append(LineWindow(start_s=line.timestamp_s, end_s=max(next_start_s, line.timestamp_s + 0.25)))
    return windows


def collect_candidate_words(
    all_words: list[TimedWord],
    window: LineWindow,
    window_pad_s: float,
) -> list[TimedWord]:
    lower_bound = max(0.0, window.start_s - window_pad_s)
    upper_bound = window.end_s + window_pad_s
    candidates: list[TimedWord] = []
    for word in all_words:
        center = (word.start_s + word.end_s) / 2.0
        if center < lower_bound:
            continue
        if center > upper_bound:
            break
        candidates.append(word)
    return candidates


def match_line_to_asr(line: LrcLine, candidate_words: list[TimedWord]) -> LineMatch:
    reference_tokens = tokenize_lyrics(line.text)
    hypothesis_tokens = [word.text for word in candidate_words]
    if not reference_tokens or not hypothesis_tokens:
        return LineMatch()

    score, alignment = semiglobal_align(reference_tokens, hypothesis_tokens)
    matched_token_map: dict[int, int] = {}
    matched_word_indices: list[int] = []
    matched_similarities: list[float] = []

    for ref_index, hyp_index, similarity in alignment:
        if ref_index is None or hyp_index is None:
            continue
        if similarity < 0.72:
            continue
        matched_token_map[ref_index] = hyp_index
        matched_word_indices.append(hyp_index)
        matched_similarities.append(similarity)

    if not matched_word_indices:
        return LineMatch(raw_score=score)

    first_word = candidate_words[min(matched_word_indices)]
    last_word = candidate_words[max(matched_word_indices)]
    confidence = sum(matched_similarities) / len(matched_similarities)
    return LineMatch(
        matched_token_map=matched_token_map,
        raw_score=score,
        confidence=confidence,
        anchor_start_s=first_word.start_s,
        anchor_end_s=last_word.end_s,
    )


def build_alignment_segments(
    lines: list[LrcLine],
    windows: list[LineWindow],
    matches: list[LineMatch],
    segment_pad_s: float,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    segments: list[dict[str, Any]] = []
    line_to_segment: dict[int, int] = {}

    for line, window, match in zip(lines, windows, matches, strict=True):
        tokens = tokenize_lyrics(line.text)
        if line.is_instrumental or not tokens:
            continue

        segment_start_s = window.start_s
        segment_end_s = window.end_s
        if match.has_anchor:
            segment_start_s = max(window.start_s, match.anchor_start_s - segment_pad_s)
            segment_end_s = min(window.end_s, match.anchor_end_s + segment_pad_s)

        if segment_end_s <= segment_start_s:
            segment_end_s = max(window.end_s, segment_start_s + 0.25)

        line_to_segment[line.index] = len(segments)
        segments.append(
            {
                "start": round(segment_start_s, 3),
                "end": round(segment_end_s, 3),
                "text": line.text,
            }
        )

    return segments, line_to_segment


def collect_expected_token_count(lines: list[LrcLine], line_to_segment: dict[int, int]) -> int:
    return sum(
        len(tokenize_lyrics(line.text))
        for line in lines
        if line.index in line_to_segment and not line.is_instrumental
    )


def collect_alignment_metrics(
    lines: list[LrcLine],
    line_to_segment: dict[int, int],
    aligned_result: Any | None,
) -> dict[str, float | int]:
    total_segments = len(line_to_segment)
    expected_tokens = collect_expected_token_count(lines, line_to_segment)
    aligned_segments = list(as_dict_value(aligned_result, "segments", []) or []) if aligned_result is not None else []

    segments_with_words = 0
    aligned_words = 0
    for line in lines:
        segment_index = line_to_segment.get(line.index)
        if segment_index is None or segment_index >= len(aligned_segments):
            continue
        raw_words = as_dict_value(aligned_segments[segment_index], "words", []) or []
        usable_word_count = 0
        for raw_word in raw_words:
            text = collapse_spaces(str(as_dict_value(raw_word, "word", as_dict_value(raw_word, "text", ""))))
            if normalize_token(text):
                usable_word_count += 1
        if usable_word_count > 0:
            segments_with_words += 1
            aligned_words += usable_word_count

    line_coverage = segments_with_words / total_segments if total_segments else 1.0
    word_coverage = aligned_words / expected_tokens if expected_tokens else 1.0
    return {
        "totalSegments": total_segments,
        "segmentsWithWords": segments_with_words,
        "expectedTokens": expected_tokens,
        "alignedWords": aligned_words,
        "lineCoverage": line_coverage,
        "wordCoverage": word_coverage,
    }


def should_accept_direct_alignment(metrics: dict[str, float | int]) -> bool:
    total_segments = int(metrics["totalSegments"])
    if total_segments == 0:
        return True
    if int(metrics["segmentsWithWords"]) == 0:
        return False
    return float(metrics["lineCoverage"]) >= 0.6 and float(metrics["wordCoverage"]) >= 0.35


def fill_missing_timings(
    tokens: list[str],
    token_timings: list[tuple[float | None, float | None]],
    segment_start_s: float,
    segment_end_s: float,
) -> list[tuple[float, float]]:
    if not tokens:
        return []

    segment_end_s = max(segment_end_s, segment_start_s)
    if not any(start is not None and end is not None for start, end in token_timings):
        step = (segment_end_s - segment_start_s) / max(len(tokens), 1)
        return [
            (segment_start_s + step * index, segment_start_s + step * (index + 1))
            for index in range(len(tokens))
        ]

    starts: list[float | None] = [start for start, _ in token_timings]
    ends: list[float | None] = [end for _, end in token_timings]
    anchors = [index for index, start in enumerate(starts) if start is not None and ends[index] is not None]

    first_anchor = anchors[0]
    if first_anchor > 0:
        left = segment_start_s
        right = float(starts[first_anchor] or segment_start_s)
        step = (right - left) / first_anchor if first_anchor else 0.0
        for index in range(first_anchor):
            starts[index] = left + step * index
            ends[index] = left + step * (index + 1)

    for left_anchor, right_anchor in zip(anchors, anchors[1:]):
        gap_count = right_anchor - left_anchor - 1
        if gap_count <= 0:
            continue
        left = float(ends[left_anchor] or starts[left_anchor] or segment_start_s)
        right = float(starts[right_anchor] or left)
        step = (right - left) / gap_count if gap_count else 0.0
        for offset, index in enumerate(range(left_anchor + 1, right_anchor), start=0):
            starts[index] = left + step * offset
            ends[index] = left + step * (offset + 1)

    last_anchor = anchors[-1]
    if last_anchor < len(tokens) - 1:
        left = float(ends[last_anchor] or starts[last_anchor] or segment_start_s)
        right = segment_end_s
        trailing_count = len(tokens) - last_anchor - 1
        step = (right - left) / trailing_count if trailing_count else 0.0
        for offset, index in enumerate(range(last_anchor + 1, len(tokens)), start=0):
            starts[index] = left + step * offset
            ends[index] = left + step * (offset + 1)

    resolved: list[tuple[float, float]] = []
    cursor = segment_start_s
    for index in range(len(tokens)):
        start_s = float(starts[index] if starts[index] is not None else cursor)
        end_s = float(ends[index] if ends[index] is not None else start_s)
        start_s = max(start_s, cursor)
        end_s = max(end_s, start_s)
        resolved.append((start_s, end_s))
        cursor = end_s

    return resolved


def map_aligned_words_to_tokens(
    line_text: str,
    aligned_words: list[TimedWord],
    segment_start_s: float,
    segment_end_s: float,
) -> list[dict[str, int | str]]:
    reference_tokens = tokenize_lyrics(line_text)
    if not reference_tokens:
        return []

    if not aligned_words:
        filled = fill_missing_timings(
            reference_tokens,
            [(None, None)] * len(reference_tokens),
            segment_start_s,
            segment_end_s,
        )
    else:
        hypothesis_tokens = [word.text for word in aligned_words]
        _, alignment = semiglobal_align(reference_tokens, hypothesis_tokens)
        provisional_timings: list[tuple[float | None, float | None]] = [(None, None)] * len(reference_tokens)
        for ref_index, hyp_index, similarity in alignment:
            if ref_index is None or hyp_index is None or similarity < 0.72:
                continue
            aligned_word = aligned_words[hyp_index]
            provisional_timings[ref_index] = (aligned_word.start_s, aligned_word.end_s)

        segment_start_s = aligned_words[0].start_s if aligned_words else segment_start_s
        segment_end_s = aligned_words[-1].end_s if aligned_words else segment_end_s
        filled = fill_missing_timings(reference_tokens, provisional_timings, segment_start_s, segment_end_s)

    result: list[dict[str, int | str]] = []
    for token, (start_s, end_s) in zip(reference_tokens, filled, strict=True):
        start_ms = max(0, round(start_s * 1000))
        end_ms = max(start_ms, round(end_s * 1000))
        result.append({"w": token, "s": start_ms, "e": end_ms})
    return result


def run_refine(model: Any, audio_path: str, result: Any, precision_s: float) -> None:
    model.refine(
        audio_path,
        result,
        precision=max(0.02, precision_s),
        word_level=True,
        inplace=True,
    )


def safe_refine_result(
    model: Any,
    audio_path: str,
    result: Any,
    precision_s: float,
    *,
    stage_label: str,
) -> Any:
    existing_words = extract_words_from_result(result)
    if not existing_words:
        print(
            f"[warn] Skipping {stage_label} refine because stable-ts returned no word-level timestamps yet.",
            file=sys.stderr,
        )
        return result

    try:
        result_copy = copy.deepcopy(result)
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(
            f"[warn] Could not clone {stage_label} result for refine; using unrefined timings instead: {exc}",
            file=sys.stderr,
        )
        return result

    try:
        run_refine(model, audio_path, result_copy, precision_s)
    except Exception as exc:  # pragma: no cover - stable-ts internal failures
        print(
            f"[warn] stable-ts refine() failed during {stage_label}; continuing with unrefined timings: {exc}",
            file=sys.stderr,
        )
        return result

    refined_words = extract_words_from_result(result_copy)
    if not refined_words:
        print(
            f"[warn] {stage_label.capitalize()} refine removed all word-level timestamps; using unrefined timings.",
            file=sys.stderr,
        )
        return result

    return result_copy


def transcribe_audio(
    model: Any,
    audio_path: str,
    language: str | None,
    min_word_dur: float,
    *,
    use_vad: bool,
    progress_callback: Callable[[float, float], None] | None = None,
) -> Any:
    return model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        verbose=None,
        regroup=False,
        suppress_silence=True,
        suppress_word_ts=True,
        vad=use_vad,
        min_word_dur=min_word_dur,
        progress_callback=progress_callback,
    )


def align_segments_to_audio(
    model: Any,
    audio_path: str,
    segments: list[dict[str, Any]],
    language: str | None,
    min_word_dur: float,
    *,
    use_vad: bool,
    progress_callback: Callable[[float, float], None] | None = None,
) -> Any:
    return model.align_words(
        audio_path,
        segments,
        language=language,
        verbose=None,
        regroup=False,
        suppress_silence=True,
        suppress_word_ts=True,
        vad=use_vad,
        min_word_dur=min_word_dur,
        progress_callback=progress_callback,
    )


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
        report_progress(
            progress_callback,
            progress_stage,
            f"Running {stage_label}.",
            percent,
            details={
                "doneSeconds": round(done_s, 2),
                "totalSeconds": round(total_s, 2),
                "segmentCount": len(segments),
            },
        )

    try:
        aligned_result = align_segments_to_audio(
            model,
            audio_path,
            segments,
            language,
            min_word_dur,
            use_vad=use_vad,
            progress_callback=stable_progress if progress_callback else None,
        )
        if not skip_refine:
            aligned_result = safe_refine_result(
                model,
                audio_path,
                aligned_result,
                refine_precision,
                stage_label=stage_label,
            )
        return aligned_result
    except Exception as exc:  # pragma: no cover - runtime fallback
        print(
            f"[warn] {stage_label} failed, falling back to the next strategy: {exc}",
            file=sys.stderr,
        )
        return None


def write_debug_outputs(
    debug_dir: Path,
    metadata: dict[str, str],
    lines: list[LrcLine],
    windows: list[LineWindow],
    matches: list[LineMatch],
    asr_words: list[TimedWord],
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "asr_words.json").write_text(
        json.dumps(
            [
                {
                    "w": word.text,
                    "s": round(word.start_s, 3),
                    "e": round(word.end_s, 3),
                    "p": word.probability,
                }
                for word in asr_words
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (debug_dir / "alignment_debug.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "lines": [
                    {
                        "index": line.index,
                        "timestamp_s": round(line.timestamp_s, 3),
                        "window_start_s": round(window.start_s, 3),
                        "window_end_s": round(window.end_s, 3),
                        "text": line.text,
                        "is_instrumental": line.is_instrumental,
                        "match_score": round(match.raw_score, 3),
                        "match_confidence": round(match.confidence, 3),
                        "anchor_start_s": None if match.anchor_start_s is None else round(match.anchor_start_s, 3),
                        "anchor_end_s": None if match.anchor_end_s is None else round(match.anchor_end_s, 3),
                    }
                    for line, window, match in zip(lines, windows, matches, strict=True)
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def safe_save_stable_json(result: Any, path: Path, label: str) -> None:
    if result is None or not hasattr(result, "save_as_json"):
        return
    try:
        result.save_as_json(str(path))
    except UnicodeEncodeError as exc:
        print(
            f"[warn] Skipping {label} debug JSON because stable-ts could not encode Vietnamese text: {exc}",
            file=sys.stderr,
        )
    except Exception as exc:  # pragma: no cover - debug output must not fail export
        print(f"[warn] Skipping {label} debug JSON: {exc}", file=sys.stderr)


def build_output_payload(
    song_title: str,
    lines: list[LrcLine],
    windows: list[LineWindow],
    line_to_segment: dict[int, int],
    aligned_result: Any | None,
) -> dict[str, Any]:
    payload_data: list[dict[str, Any]] = []
    aligned_segments = list(as_dict_value(aligned_result, "segments", []) or []) if aligned_result is not None else []

    for line, window in zip(lines, windows, strict=True):
        if line.is_instrumental or not tokenize_lyrics(line.text):
            payload_data.append(
                {
                    "line": line.text,
                    "startTimeMs": max(0, round(line.timestamp_s * 1000)),
                    "words": [],
                }
            )
            continue

        segment_index = line_to_segment.get(line.index)
        segment = aligned_segments[segment_index] if segment_index is not None and segment_index < len(aligned_segments) else None
        segment_words = []
        if segment is not None:
            for word in as_dict_value(segment, "words", []) or []:
                text = collapse_spaces(str(as_dict_value(word, "word", as_dict_value(word, "text", ""))))
                if not normalize_token(text):
                    continue
                segment_words.append(
                    TimedWord(
                        text=text,
                        start_s=float(as_dict_value(word, "start", window.start_s) or window.start_s),
                        end_s=float(as_dict_value(word, "end", window.start_s) or window.start_s),
                    )
                )

        mapped_words = map_aligned_words_to_tokens(line.text, segment_words, window.start_s, window.end_s)
        start_time_ms = mapped_words[0]["s"] if mapped_words else max(0, round(line.timestamp_s * 1000))
        payload_data.append(
            {
                "line": line.text,
                "startTimeMs": start_time_ms,
                "words": mapped_words,
            }
        )

    return {"song": song_title, "data": payload_data}


def report_progress(
    progress_callback: ProgressCallback | None,
    stage: str,
    message: str,
    percent: int,
    *,
    details: dict[str, Any] | None = None,
) -> None:
    if progress_callback is None:
        return
    progress_callback(stage, message, percent, details)


def run_alignment(
    audio_path: str | Path,
    lrc_path: str | Path,
    output_path: str | Path,
    *,
    song: str | None = None,
    model_name: str = "recommended",
    device: str = "auto",
    language: str | None = None,
    window_pad_ms: int = 300,
    segment_pad_ms: int = 180,
    tail_pad_ms: int = 4000,
    min_word_dur: float = 0.02,
    refine_precision: float = 0.02,
    skip_refine: bool = False,
    strategy: str = "auto",
    use_vad: bool = True,
    debug_dir: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    audio_path = Path(audio_path).expanduser().resolve()
    lrc_path = Path(lrc_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    report_progress(progress_callback, "validating", "Validating input files.", 2)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not lrc_path.is_file():
        raise FileNotFoundError(f"LRC file not found: {lrc_path}")

    metadata, lrc_lines = parse_lrc(lrc_path)
    resolved_device = resolve_device_choice(device)
    resolved_model_name = resolve_model_choice(model_name, resolved_device)
    report_progress(
        progress_callback,
        "loading_model",
        f"Loading model {resolved_model_name} on {resolved_device}.",
        8,
        details={"resolvedModel": resolved_model_name, "resolvedDevice": resolved_device},
    )
    model = get_alignment_model(resolved_model_name, resolved_device)
    report_progress(
        progress_callback,
        "preparing_segments",
        "Preparing alignment windows from the LRC timestamps.",
        16,
        details={"lineCount": len(lrc_lines)},
    )

    direct_inferred_end_s = lrc_lines[-1].timestamp_s + tail_pad_ms / 1000.0
    direct_windows = build_line_windows(lrc_lines, direct_inferred_end_s, tail_pad_ms / 1000.0)
    direct_matches = [LineMatch() for _ in lrc_lines]
    direct_segments, direct_line_to_segment = build_alignment_segments(
        lrc_lines,
        direct_windows,
        direct_matches,
        segment_pad_ms / 1000.0,
    )

    raw_result = None
    asr_words: list[TimedWord] = []
    windows = direct_windows
    matches = direct_matches
    line_to_segment = direct_line_to_segment
    aligned_result = None
    strategy_used = "direct"

    if strategy in ("auto", "direct"):
        report_progress(progress_callback, "direct_alignment", "Running direct forced alignment from LRC timestamps.", 28)
        aligned_result = run_forced_alignment(
            model,
            str(audio_path),
            direct_segments,
            language,
            min_word_dur,
            refine_precision,
            skip_refine,
            stage_label="direct forced alignment",
            use_vad=use_vad,
            progress_callback=progress_callback,
            progress_stage="direct_alignment",
            progress_start=28,
            progress_end=44,
        )
        direct_metrics = collect_alignment_metrics(lrc_lines, direct_line_to_segment, aligned_result)
        report_progress(
            progress_callback,
            "direct_alignment_review",
            "Evaluating direct alignment coverage.",
            44,
            details=direct_metrics,
        )
        if aligned_result is not None and (strategy == "direct" or should_accept_direct_alignment(direct_metrics)):
            windows = direct_windows
            matches = direct_matches
            line_to_segment = direct_line_to_segment
            report_progress(
                progress_callback,
                "direct_alignment_accepted",
                "Direct alignment coverage is good enough. Skipping slower transcription.",
                58,
                details=direct_metrics,
            )
        else:
            aligned_result = None
            if strategy == "auto":
                strategy_used = "transcribe"
                print(
                    "[warn] Direct LRC forced alignment coverage was too low; switching to transcription-assisted alignment.",
                    file=sys.stderr,
                )
                report_progress(
                    progress_callback,
                    "transcription_fallback",
                    "Direct alignment coverage was low. Switching to transcription-assisted alignment.",
                    58,
                    details=direct_metrics,
                )
            else:
                windows = direct_windows
                matches = direct_matches
                line_to_segment = direct_line_to_segment

    if aligned_result is None and strategy in ("auto", "transcribe"):
        report_progress(progress_callback, "transcribing", "Running stable-ts transcription to improve matching.", 66)
        last_transcribe_percent = -1

        def transcribe_progress(done_s: float, total_s: float) -> None:
            nonlocal last_transcribe_percent
            if total_s <= 0:
                return
            ratio = min(max(done_s / total_s, 0.0), 1.0)
            percent = 66 + round(6 * ratio)
            if percent <= last_transcribe_percent:
                return
            last_transcribe_percent = percent
            report_progress(
                progress_callback,
                "transcribing",
                "Running stable-ts transcription to improve matching.",
                percent,
                details={"doneSeconds": round(done_s, 2), "totalSeconds": round(total_s, 2)},
            )

        raw_result = transcribe_audio(
            model,
            str(audio_path),
            language,
            min_word_dur,
            use_vad=use_vad,
            progress_callback=transcribe_progress if progress_callback else None,
        )

        if not skip_refine:
            report_progress(progress_callback, "refining_transcript", "Refining the initial transcript.", 72)
            raw_result = safe_refine_result(
                model,
                str(audio_path),
                raw_result,
                refine_precision,
                stage_label="initial transcription",
            )

        asr_words = extract_words_from_result(raw_result)
        raw_segments = extract_segment_ranges_from_result(raw_result)
        inferred_audio_end_s = infer_result_end_s(
            raw_result,
            asr_words,
            raw_segments,
            direct_inferred_end_s,
        )
        windows = build_line_windows(lrc_lines, inferred_audio_end_s, tail_pad_ms / 1000.0)

        if asr_words:
            report_progress(progress_callback, "matching_transcript", "Matching transcript words to the LRC lines.", 78)
            matches = [
                LineMatch() if line.is_instrumental else match_line_to_asr(
                    line,
                    collect_candidate_words(asr_words, window, window_pad_ms / 1000.0),
                )
                for line, window in zip(lrc_lines, windows, strict=True)
            ]
        else:
            print(
                "[warn] Initial stable-ts transcription returned no word-level timestamps; using plain LRC time windows.",
                file=sys.stderr,
            )
            matches = [LineMatch() for _ in lrc_lines]
            report_progress(
                progress_callback,
                "plain_windows",
                "Transcript had no word-level timestamps. Falling back to plain LRC windows.",
                78,
            )

        transcription_segments, line_to_segment = build_alignment_segments(
            lrc_lines,
            windows,
            matches,
            segment_pad_ms / 1000.0,
        )

        report_progress(progress_callback, "forced_alignment", "Running transcription-assisted forced alignment.", 86)
        aligned_result = run_forced_alignment(
            model,
            str(audio_path),
            transcription_segments,
            language,
            min_word_dur,
            refine_precision,
            skip_refine,
            stage_label="transcription-assisted forced alignment",
            use_vad=use_vad,
            progress_callback=progress_callback,
            progress_stage="forced_alignment",
            progress_start=86,
            progress_end=94,
        )

    song_title = song or metadata.get("ti") or audio_path.stem
    report_progress(progress_callback, "building_output", "Building lyrics_pro.json payload.", 94)
    payload = build_output_payload(song_title, lrc_lines, windows, line_to_segment, aligned_result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if debug_dir:
        debug_path = Path(debug_dir).expanduser().resolve()
        write_debug_outputs(debug_path, metadata, lrc_lines, windows, matches, asr_words)
        (debug_path / "run_meta.json").write_text(
            json.dumps(
                {
                    "requestedModel": model_name,
                    "resolvedModel": resolved_model_name,
                    "requestedDevice": device,
                    "resolvedDevice": resolved_device,
                    "strategyRequested": strategy,
                    "strategyUsed": strategy_used,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        safe_save_stable_json(aligned_result, debug_path / "stable_aligned_segments.json", "aligned segments")
        safe_save_stable_json(raw_result, debug_path / "stable_raw_transcript.json", "raw transcript")

    report_progress(
        progress_callback,
        "completed",
        "Alignment completed. Export file is ready.",
        100,
        details={"outputPath": str(output_path), "song": song_title},
    )
    return payload


def main() -> int:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    run_alignment(
        args.audio,
        args.lrc,
        output_path,
        song=args.song,
        model_name=args.model,
        device=args.device,
        language=args.language,
        window_pad_ms=args.window_pad_ms,
        segment_pad_ms=args.segment_pad_ms,
        tail_pad_ms=args.tail_pad_ms,
        min_word_dur=args.min_word_dur,
        refine_precision=args.refine_precision,
        skip_refine=args.skip_refine,
        strategy=args.strategy,
        use_vad=True,
        debug_dir=args.debug_dir,
    )

    print(f"Wrote aligned lyric data to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
