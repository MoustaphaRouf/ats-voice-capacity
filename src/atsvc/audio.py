"""Speech segmentation (energy VAD) and acoustic feature extraction.

A TopSky recording of an executive (radio) position mixes controller speech,
pilot speech and channel noise. Each recording is first cut into continuous
speech segments by an energy-based voice-activity detector (VAD); each segment
is then described by a fixed-length acoustic feature vector that the
classifier (see :mod:`atsvc.classifier`) labels CTL / PILOT / NOISE.

The controller is recorded directly at the console microphone, whereas pilots
arrive through the VHF air-ground link (band-limited, noisier, with carrier
artefacts). The features below were chosen to capture that difference:
timbre (MFCC), spectral shape (centroid, bandwidth, roll-off), tonality (ZCR),
level (RMS) and an in-segment SNR estimate.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_MFCC = 13
FRAME_S = 0.02  # VAD analysis frame
HOP_S = 0.01  # VAD hop
NOISE_PERCENTILE = 20  # frames below this RMS percentile define the noise floor
THRESHOLD_FACTOR = 3.0  # speech = RMS above 3 x noise floor
MIN_FEATURE_DURATION_S = 0.2


@dataclass
class Segment:
    """One continuous speech segment inside a recording."""

    start_s: float
    end_s: float
    audio: np.ndarray
    sr: int

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def load_audio(path: str | Path, sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load any audio file as mono float32, resampled to ``sr``."""
    import librosa

    y, sr_out = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32), sr_out


def frame_rms(y: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """RMS energy on short frames; returns (rms per frame, hop in samples)."""
    frame = max(int(FRAME_S * sr), 1)
    hop = max(int(HOP_S * sr), 1)
    if len(y) < frame:
        return np.array([np.sqrt(np.mean(y**2))]) if len(y) else np.zeros(1), hop
    n = (len(y) - frame) // hop + 1
    # Strided view avoids the Python loop of the original notebook.
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return np.sqrt(np.mean(y[idx] ** 2, axis=1)), hop


def speech_mask(rms: np.ndarray) -> np.ndarray:
    """Boolean speech/non-speech decision per frame (noise-floor threshold)."""
    threshold = np.percentile(rms, NOISE_PERCENTILE) * THRESHOLD_FACTOR
    return rms > threshold


def segment_vad(y: np.ndarray, sr: int, min_duration_s: float = 0.5) -> list[Segment]:
    """Cut a recording into continuous speech segments.

    Each segment is, in principle, a single transmission by a single speaker,
    which makes it the correct unit for labelling multi-speaker channels.

    Note: runs are detected on a mask padded with ``False`` at both ends, so a
    recording that starts or ends in the middle of a transmission is handled
    correctly (the original notebook paired rising and falling edges by index,
    which shifted every segment by one when the file began with speech).
    """
    rms, hop = frame_rms(y, sr)
    mask = speech_mask(rms)
    edges = np.diff(np.concatenate([[False], mask, [False]]).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    segments = []
    for f0, f1 in zip(starts, ends):
        t0, t1 = f0 * hop / sr, f1 * hop / sr
        if t1 - t0 < min_duration_s:
            continue
        audio = y[int(t0 * sr) : int(t1 * sr)]
        segments.append(Segment(round(t0, 3), round(t1, 3), audio, sr))
    return segments


def segment_snr_db(y: np.ndarray, sr: int) -> float:
    """SNR estimate: power of speech frames over power of noise frames."""
    rms, _ = frame_rms(y, sr)
    mask = speech_mask(rms)
    p_sig = np.mean(rms[mask] ** 2) if mask.any() else 1e-10
    p_noise = np.mean(rms[~mask] ** 2) if (~mask).any() else 1e-10
    return float(10 * np.log10(max(p_sig, 1e-12) / max(p_noise, 1e-12)))


def extract_features(y: np.ndarray, sr: int) -> dict | None:
    """Fixed-length acoustic description of one segment (36 features).

    Returns ``None`` for segments too short to describe reliably. A single
    function serves both whole files and in-memory segments (the original
    code had two near-identical copies of this logic).
    """
    import librosa

    duration = len(y) / sr
    if duration < MIN_FEATURE_DURATION_S:
        return None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]

    feats = {
        "duration_s": duration,
        "snr_db": segment_snr_db(y, sr),
        "centroid_mean": centroid.mean(),
        "centroid_std": centroid.std(),
        "bandwidth_mean": bandwidth.mean(),
        "bandwidth_std": bandwidth.std(),
        "rolloff_mean": rolloff.mean(),
        "zcr_mean": zcr.mean(),
        "rms_mean": rms.mean(),
        "rms_std": rms.std(),
    }
    for i in range(N_MFCC):
        feats[f"mfcc{i}_mean"] = mfcc[i].mean()
        feats[f"mfcc{i}_std"] = mfcc[i].std()
    return {k: float(v) for k, v in feats.items()}


FEATURE_COLUMNS = (
    ["duration_s", "snr_db", "centroid_mean", "centroid_std", "bandwidth_mean",
     "bandwidth_std", "rolloff_mean", "zcr_mean", "rms_mean", "rms_std"]
    + [f"mfcc{i}_{s}" for i in range(N_MFCC) for s in ("mean", "std")]
)
