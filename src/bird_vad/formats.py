from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any


def _to_json_safe(obj: Any) -> Any:
    """
    Convert common Python objects to JSON-serializable forms.
    """
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    return obj


def write_vad_json(
    out_path: Path,
    *,
    device_id: str,
    chunk_id: str,
    source_wav: Path,
    vad_wav: Path,
    intervals: List[Tuple[float, float]],
    clip_filenames: List[str],
    model_info: Dict[str, Any],
    extra_meta: Dict[str, Any] | None = None,
) -> None:
    """
    Write the canonical VAD result JSON.

    This is the file you upload.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema": "bird-vad.v1",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "device_id": device_id,
        "chunk_id": chunk_id,
        "audio": {
            "source_wav": str(source_wav),
            "vad_wav": str(vad_wav),
        },
        "segments": [
            {
                "start_s": float(start),
                "end_s": float(end),
                "label": "speech",
            }
            for start, end in intervals
        ],
        "clips": clip_filenames,
        "model": model_info,
    }

    if extra_meta:
        payload["meta"] = _to_json_safe(extra_meta)

    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_rttm(
    out_path: Path,
    *,
    recording_id: str,
    intervals: List[Tuple[float, float]],
    channel: int = 1,
    speaker: str = "speech",
) -> None:
    """
    Optional: write RTTM format (standard in speech research).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    for start, end in intervals:
        dur = max(0.0, end - start)
        lines.append(
            f"SPEAKER {recording_id} {channel} "
            f"{start:.3f} {dur:.3f} <NA> <NA> {speaker} <NA> <NA>"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(
    out_path: Path,
    *,
    intervals: List[Tuple[float, float]],
) -> None:
    """
    Optional: simple CSV for quick inspection.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["start_s,end_s"]
    for s, e in intervals:
        lines.append(f"{s:.6f},{e:.6f}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
