# src/bird_vad/vad_javad.py
#
# JaVAD wrapper assuming JaVAD is installed as a dependency, e.g.:
#   pip install git+https://github.com/<ORG>/<REPO>.git
# or (dev mode):
#   pip install -e /path/to/javad-main
#
# Then you can import:
#   from javad.extras.basic import get_speech_intervals

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass(frozen=True)
class JaVADSettings:
    checkpoint_path: str                 # env: JAVAD_CHECKPOINT (path to .pt)
    model_name: str = "balanced"         # env: JAVAD_MODEL_NAME (tiny/balanced/precise)
    device: str = "cpu"                  # Pi = cpu

    # Post-processing knobs (simple + effective)
    min_speech_ms: int = 200             # drop speech segments shorter than this
    min_silence_ms: int = 200            # merge segments separated by <= this gap


def _merge_close(intervals: List[Tuple[float, float]], max_gap_s: float) -> List[Tuple[float, float]]:
    """Merge intervals if the gap between them is <= max_gap_s."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged: List[Tuple[float, float]] = [intervals[0]]

    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe + max_gap_s:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _drop_short(intervals: List[Tuple[float, float]], min_len_s: float) -> List[Tuple[float, float]]:
    """Drop intervals shorter than min_len_s."""
    return [(s, e) for (s, e) in intervals if (e - s) >= min_len_s]


def run_javad_vad(audio_path: str | Path, settings: JaVADSettings) -> List[Tuple[float, float]]:
    """
    Runs JaVAD on a WAV (or audio file) and returns post-processed speech intervals.

    Returns:
        List of (start_seconds, end_seconds)
    """
    audio_path = Path(audio_path)

    # ✅ Dependency import (clean, no vendoring)

from javad.extras import get_speech_intervals

raw = get_speech_intervals(
    str(audio_path),
    checkpoint=settings.checkpoint_path,   # <-- THIS is the key
    model_name=settings.model_name,        # optional if supported
    device=settings.device,                # optional if supported
    )

    # from javad.extras import get_speech_intervals

    # raw = get_speech_intervals(str(audio_path))
    # intervals = [(float(s), float(e)) for (s, e) in raw]

    # raw = get_speech_intervals(
    #     audio=str(audio_path),
    #     checkpoint=settings.checkpoint_path,
    #     model_name=settings.model_name,
    #     device=settings.device,
    # )

    # intervals: List[Tuple[float, float]] = [(float(s), float(e)) for (s, e) in raw]

    # Post-process
max_gap_s = max(0.0, settings.min_silence_ms / 1000.0)
min_len_s = max(0.0, settings.min_speech_ms / 1000.0)

intervals = _merge_close(intervals, max_gap_s=max_gap_s)
intervals = _drop_short(intervals, min_len_s=min_len_s)

return intervals
