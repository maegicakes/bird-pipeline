from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

from javad.extras import get_speech_intervals


@dataclass(frozen=True)
class JaVADSettings:
    checkpoint_path: str
    model_name: str = "balanced"
    device: str = "cpu"
    min_speech_ms: int = 200
    min_silence_ms: int = 200


def _merge_close(intervals: List[Tuple[float, float]], max_gap_s: float):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe + max_gap_s:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _drop_short(intervals: List[Tuple[float, float]], min_len_s: float):
    return [(s, e) for s, e in intervals if (e - s) >= min_len_s]


def run_javad_vad(audio_path: str | Path, settings: JaVADSettings):
    audio_path = Path(audio_path)

    # 🔑 THIS is the correct JaVAD call (checkpoint REQUIRED)
    # raw = get_speech_intervals(
    #     str(audio_path),
    #     checkpoint=settings.checkpoint_path,
    # )

    # # convert to floats
    # intervals: List[Tuple[float, float]] = [
    #     (float(s), float(e)) for (s, e) in raw
    # ]

    raw = get_speech_intervals(
    str(audio_path),
    checkpoint=settings.checkpoint_path,
    )

# JaVAD returns either:
# 1) intervals directly: [(start, end), ...]
# 2) (frame_scores, intervals): (list[float], list[tuple])
    if isinstance(raw, tuple) and len(raw) == 2:
        raw_intervals = raw[1]
    else:
        raw_intervals = raw

    intervals = []
    for item in raw_intervals:
        # item could be (start, end) or (start, end, score, ...)
        s = float(item[0])
        e = float(item[1])
        intervals.append((s, e))

    # post-processing
    max_gap_s = settings.min_silence_ms / 1000.0
    min_len_s = settings.min_speech_ms / 1000.0

    intervals = _merge_close(intervals, max_gap_s)
    intervals = _drop_short(intervals, min_len_s)

    return intervals
