from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RecorderSettings:
    arecord_device: str
    rate_hz: int = 44100
    channels: int = 2
    duration_s: int = 60
    audio_dir: Path = Path("./data_temp/Audios")


def _timestamp_name(prefix: str = "", suffix: str = ".wav") -> str:
    # Matches the typical bird-files timestamp style
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{prefix}{ts}{suffix}"


def record_wav_chunk(settings: RecorderSettings, filename: str | None = None) -> Path:
    """
    Record one WAV chunk using `arecord`.

    This is based on the approach in bird-files-main:
      - records fixed-length chunks
      - device comes from env/config (ARECORD_DEVICE)
      - stores in an audio directory
    """
    settings.audio_dir.mkdir(parents=True, exist_ok=True)

    name = filename or _timestamp_name()
    out_wav = settings.audio_dir / name

    # Write to a temp file first, then rename (avoids half-written files if interrupted)
    tmp_wav = out_wav.with_suffix(out_wav.suffix + ".tmp")

    cmd = [
        "arecord",
        "--device", settings.arecord_device,
        "--rate", str(settings.rate_hz),
        "--format", "S16_LE",
        "--channels", str(settings.channels),
        "--duration", str(settings.duration_s),
        str(tmp_wav),
    ]

    # This will raise CalledProcessError if arecord fails
    subprocess.run(cmd, check=True)

    tmp_wav.replace(out_wav)
    return out_wav
