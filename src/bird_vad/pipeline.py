# src/bird_vad/pipeline.py

from __future__ import annotations

import os
import subprocess
import time
import glob
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import soundfile as sf

from bird_vad.config import load_config
from bird_vad.recorder import RecorderSettings, record_wav_chunk
from bird_vad.vad_javad import JaVADSettings, run_javad_vad
from bird_vad.formats import write_vad_json
from bird_vad.uploader import (
    load_s3_settings_from_env,
    upload_json_result,
    upload_wav_clips_for_chunk,
)


# ----------------------------
# Audio helpers
# ----------------------------

def _cmd_exists(cmd: str) -> bool:
    return subprocess.call(["which", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def convert_wav_for_vad(
    in_wav: Path,
    out_wav: Path,
    *,
    target_sr: int = 16000,
    target_channels: int = 1,
) -> Path:
    """
    Convert to JaVAD-friendly wav (typically 16kHz mono).
    Prefer ffmpeg if available, fallback to librosa+soundfile.
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    if _cmd_exists("ffmpeg"):
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(in_wav),
            "-ac",
            str(target_channels),
            "-ar",
            str(target_sr),
            str(out_wav),
        ]
        subprocess.run(cmd, check=True)
        return out_wav

    # Fallback: librosa
    import librosa  # local import

    y, _sr = librosa.load(str(in_wav), sr=target_sr, mono=(target_channels == 1))
    sf.write(str(out_wav), y, target_sr, subtype="PCM_16")
    return out_wav


def clip_segments_to_wavs(
    wav_path: Path,
    segments: List[Tuple[float, float]],
    *,
    clips_dir: Path,
    chunk_id: str,
    max_clip_s: Optional[float] = None,
    min_clip_s: float = 0.0,
) -> List[Path]:
    """
    Write one wav per speech segment as:
      <chunk_id>_000.wav, <chunk_id>_001.wav, ...
    """
    clips_dir.mkdir(parents=True, exist_ok=True)

    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)

    out_paths: List[Path] = []
    for idx, (s, e) in enumerate(segments):
        if e <= s:
            continue

        # cap clip length if requested
        if max_clip_s is not None and (e - s) > max_clip_s:
            e = s + max_clip_s

        if (e - s) < min_clip_s:
            continue

        start_i = int(round(s * sr))
        end_i = int(round(e * sr))
        start_i = max(0, min(start_i, len(audio)))
        end_i = max(0, min(end_i, len(audio)))
        if end_i <= start_i:
            continue

        clip = audio[start_i:end_i]
        out = clips_dir / f"{chunk_id}_{idx:03d}.wav"
        sf.write(str(out), clip, sr, subtype="PCM_16")
        out_paths.append(out)

    return out_paths


# ----------------------------
# “Old pipeline” style cleanup
# ----------------------------

def _delete_sidecars_for_source_wav(source_wav: Path) -> None:
    """
    Delete any files that share the same base name as source_wav.
    Similar idea to old: glob(file.replace(".wav","*")) then remove.
    """
    pattern = str(source_wav).replace(".wav", "*")
    for p in glob.glob(pattern):
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass


# ----------------------------
# Core processing (shared)
# ----------------------------

def process_one_audio(
    *,
    cfg,
    jcfg: JaVADSettings,
    s3_settings,
    chunk_id: str,
    source_wav: Path,
    clips_dir: Path,
    vad_sr: int,
    vad_ch: int,
    upload_audio_clips: bool,
    max_clip_s: Optional[float],
    min_clip_s: float,
    upload_enabled: bool,
    delete_source_wav: bool,
    delete_vad_wav: bool,
    delete_clips: bool,
) -> None:
    """
    Process exactly one source wav:
      convert -> JaVAD -> clips -> write JSON -> upload -> cleanup
    Works for BOTH record mode and watch mode.
    """
    vad_wav: Optional[Path] = None
    out_json: Optional[Path] = None
    clip_paths: List[Path] = []

    # 1) convert to JaVAD-friendly audio
    vad_wav = cfg.recorder.audio_dir / f"{chunk_id}.vad_input.wav"
    convert_wav_for_vad(source_wav, vad_wav, target_sr=vad_sr, target_channels=vad_ch)

    # 2) run JaVAD (timestamps)
    print(f"[vad] {vad_wav.name}")
    intervals = run_javad_vad(vad_wav, jcfg)

    # 3) clip audio segments (optional)
    if upload_audio_clips and intervals:
        print(f"[clip] segments={len(intervals)}")
        clip_paths = clip_segments_to_wavs(
            vad_wav,
            intervals,
            clips_dir=clips_dir,
            chunk_id=chunk_id,
            max_clip_s=max_clip_s,
            min_clip_s=min_clip_s,
        )

    # 4) write JSON (canonical output)
    out_json = cfg.vad.results_dir / f"{chunk_id}.vad.json"

    model_info = {
        "name": "javad",
        "model_name": jcfg.model_name,
        "checkpoint": jcfg.checkpoint_path,
        "device": jcfg.device,
        "postprocess": {
            "min_speech_ms": jcfg.min_speech_ms,
            "min_silence_ms": jcfg.min_silence_ms,
            "threshold": cfg.vad.threshold,  # informational
        },
    }
    extra_meta = {
        "recording": {
            "arecord_device": cfg.recorder.arecord_device,
            "rate_hz": cfg.recorder.rate_hz,
            "channels": cfg.recorder.channels,
            "duration_s": cfg.recorder.duration_s,
        },
        "vad_input": {"sample_rate": vad_sr, "channels": vad_ch},
        "clips": {
            "enabled": upload_audio_clips,
            "min_clip_s": min_clip_s,
            "max_clip_s": max_clip_s,
        },
    }

    write_vad_json(
        out_json,
        device_id=cfg.device_id,
        chunk_id=chunk_id,
        source_wav=source_wav,
        vad_wav=vad_wav,
        intervals=intervals,
        clip_filenames=[p.name for p in clip_paths],
        model_info=model_info,
        extra_meta=extra_meta,
    )

    print(f"[result] {out_json.name} segments={len(intervals)} clips={len(clip_paths)}")

    # 5) upload JSON (+ clips)
    if upload_enabled and s3_settings is not None:
        print(f"[upload] json {out_json.name}")
        upload_json_result(out_json, settings=s3_settings)

        if upload_audio_clips and clip_paths:
            print(f"[upload] clips {chunk_id} ({len(clip_paths)})")
            upload_wav_clips_for_chunk(clips_dir, chunk_id, settings=s3_settings)

    # 6) optional cleanup
    if delete_source_wav:
        source_wav.unlink(missing_ok=True)
    if delete_vad_wav and vad_wav:
        vad_wav.unlink(missing_ok=True)
    if delete_clips and clip_paths:
        for p in clip_paths:
            p.unlink(missing_ok=True)


# ----------------------------
# Watch mode (old upload_process.py behavior)
# ----------------------------

def watch_loop(
    *,
    cfg,
    jcfg: JaVADSettings,
    s3_settings,
    clips_dir: Path,
    vad_sr: int,
    vad_ch: int,
    upload_audio_clips: bool,
    max_clip_s: Optional[float],
    min_clip_s: float,
    upload_enabled: bool,
    delete_source_wav: bool,
    delete_vad_wav: bool,
    delete_clips: bool,
) -> int:
    """
    Watch cfg.recorder.audio_dir for .wav files.
    Skip newest file (still being written), process the rest.
    Optionally delete sidecars like old script.
    """
    delete_sidecars = os.getenv("DELETE_SIDECARS", "1").strip().lower() in {"1", "true", "yes", "y", "on"}
    sleep_s = float(os.getenv("WATCH_SLEEP_SECONDS", "1"))

    print("[watch] mode enabled")
    print(f"[watch] watching: {cfg.recorder.audio_dir}")
    print(f"[watch] delete_sidecars={delete_sidecars}")

    while True:
        try:
            wavs = sorted([p for p in cfg.recorder.audio_dir.glob("*.wav")])

            # If only 0/1 wav exists, likely still recording or nothing to do
            if len(wavs) <= 1:
                time.sleep(sleep_s)
                continue

            # Skip the newest file (might still be open)
            to_process = wavs[:-1]

            for source_wav in to_process:
                chunk_id = source_wav.stem
                print(f"[watch] process {source_wav.name}")

                process_one_audio(
                    cfg=cfg,
                    jcfg=jcfg,
                    s3_settings=s3_settings,
                    chunk_id=chunk_id,
                    source_wav=source_wav,
                    clips_dir=clips_dir,
                    vad_sr=vad_sr,
                    vad_ch=vad_ch,
                    upload_audio_clips=upload_audio_clips,
                    max_clip_s=max_clip_s,
                    min_clip_s=min_clip_s,
                    upload_enabled=upload_enabled,
                    delete_source_wav=delete_source_wav,
                    delete_vad_wav=delete_vad_wav,
                    delete_clips=delete_clips,
                )

                if delete_sidecars:
                    _delete_sidecars_for_source_wav(source_wav)

            time.sleep(sleep_s)

        except KeyboardInterrupt:
            print("[watch] stopping (Ctrl+C)")
            return 0
        except subprocess.CalledProcessError as e:
            print(f"[watch][error] subprocess failed: {e}")
            time.sleep(1)
        except Exception as e:
            print(f"[watch][error] {type(e).__name__}: {e}")
            time.sleep(1)


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    cfg = load_config()

    # --- Required: JaVAD checkpoint path ---
    checkpoint_path = os.getenv("JAVAD_CHECKPOINT", "").strip()
    if not checkpoint_path:
        raise RuntimeError("JAVAD_CHECKPOINT env var is required (path to JaVAD .pt checkpoint).")

    model_name = os.getenv("JAVAD_MODEL_NAME", "balanced").strip()
    device = os.getenv("JAVAD_DEVICE", "cpu").strip()  # keep as cpu for Pi

    # --- VAD input format ---
    vad_sr = int(os.getenv("VAD_SAMPLE_RATE", "16000"))
    vad_ch = int(os.getenv("VAD_CHANNELS", "1"))

    # --- Mode (record/watch) ---
    mode = os.getenv("PIPELINE_MODE", "record").strip().lower()

    # --- Clipping options ---
    upload_audio_clips = os.getenv("UPLOAD_AUDIO_CLIPS", "1").strip().lower() in {"1", "true", "yes", "y", "on"}
    clips_dir = Path(os.getenv("CLIPS_DIR", "./data_temp/Clips")).expanduser().resolve()
    clips_dir.mkdir(parents=True, exist_ok=True)

    max_clip_s_env = os.getenv("MAX_CLIP_SECONDS", "").strip()
    max_clip_s = float(max_clip_s_env) if max_clip_s_env else None
    min_clip_s = float(os.getenv("MIN_CLIP_SECONDS", str(cfg.vad.min_speech_ms / 1000.0)).strip())

    # --- Cleanup options ---
    delete_source_wav = os.getenv("DELETE_SOURCE_WAV", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
    delete_vad_wav = os.getenv("DELETE_VAD_WAV", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
    delete_clips = os.getenv("DELETE_CLIPS_LOCAL", "0").strip().lower() in {"1", "true", "yes", "y", "on"}

    # --- Upload ---
    upload_enabled = bool(cfg.upload.enabled)
    s3_settings = load_s3_settings_from_env() if upload_enabled else None

    # --- Recorder settings (bird-files style) ---
    rec = RecorderSettings(
        arecord_device=cfg.recorder.arecord_device,
        rate_hz=cfg.recorder.rate_hz,
        channels=cfg.recorder.channels,
        duration_s=cfg.recorder.duration_s,
        audio_dir=cfg.recorder.audio_dir,
    )

    # --- JaVAD settings ---
    jcfg = JaVADSettings(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        device=device,
        min_speech_ms=cfg.vad.min_speech_ms,
        min_silence_ms=cfg.vad.min_silence_ms,
    )

    print("[pipeline] starting")
    print(f"[pipeline] mode={mode}")
    print(f"[pipeline] device_id={cfg.device_id}")
    print(f"[pipeline] audio_dir={cfg.recorder.audio_dir}")
    print(f"[pipeline] results_dir={cfg.vad.results_dir}")
    print(f"[pipeline] clips_dir={clips_dir}")
    print(f"[pipeline] upload_enabled={upload_enabled} upload_audio_clips={upload_audio_clips}")
    print(f"[pipeline] javad model={jcfg.model_name} device={jcfg.device}")
    print(f"[pipeline] vad input target: {vad_sr} Hz, ch={vad_ch}")

    if mode == "watch":
        return watch_loop(
            cfg=cfg,
            jcfg=jcfg,
            s3_settings=s3_settings,
            clips_dir=clips_dir,
            vad_sr=vad_sr,
            vad_ch=vad_ch,
            upload_audio_clips=upload_audio_clips,
            max_clip_s=max_clip_s,
            min_clip_s=min_clip_s,
            upload_enabled=upload_enabled,
            delete_source_wav=delete_source_wav,
            delete_vad_wav=delete_vad_wav,
            delete_clips=delete_clips,
        )

    # ----------------------------
    # Record mode (current behavior)
    # ----------------------------
    while True:
        chunk_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        source_wav: Optional[Path] = None
        try:
            # 1) record
            print(f"[record] {chunk_id}")
            source_wav = record_wav_chunk(rec, filename=f"{chunk_id}.wav")

            # 2+) process (convert -> vad -> clips -> json -> upload -> cleanup)
            process_one_audio(
                cfg=cfg,
                jcfg=jcfg,
                s3_settings=s3_settings,
                chunk_id=chunk_id,
                source_wav=source_wav,
                clips_dir=clips_dir,
                vad_sr=vad_sr,
                vad_ch=vad_ch,
                upload_audio_clips=upload_audio_clips,
                max_clip_s=max_clip_s,
                min_clip_s=min_clip_s,
                upload_enabled=upload_enabled,
                delete_source_wav=delete_source_wav,
                delete_vad_wav=delete_vad_wav,
                delete_clips=delete_clips,
            )

        except KeyboardInterrupt:
            print("[pipeline] stopping (Ctrl+C)")
            return 0
        except subprocess.CalledProcessError as e:
            print(f"[error] subprocess failed: {e}")
        except Exception as e:
            print(f"[error] {type(e).__name__}: {e}")

        time.sleep(1)


if __name__ == "__main__":
    raise SystemExit(main())
