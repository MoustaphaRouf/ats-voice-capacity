"""Synthetic data with known ground truth.

Real TopSky recordings and traffic statistics cannot be published, so the
repository ships two generators:

* :func:`simulate_operations` — a month of approach-sector operations:
  TMA entries with a daily traffic profile, and for each flight a sequence of
  pilot/controller exchanges on a single half-duplex frequency (one speaker at
  a time: an exchange waits while the frequency is busy). Its output has the
  same columns as the classifier's predictions, including a controlled rate
  of misclassified and noise segments. Because the radio service per flight
  is drawn from known distributions, it lets us check that the pipeline
  recovers d_radio.
* :func:`make_audio_corpus` — short WAV recordings mixing synthetic
  "controller" speech (console microphone, full band), "pilot" speech
  (VHF-like: band-limited, noisier) and channel noise (carrier beeps, bursts).
  It exercises the classifier end to end. It is a mechanics test, not a
  substitute for real speech: accuracy on it says nothing about accuracy on
  real recordings.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt

# ─────────────────────────────────────────────────────────────────────────────
# Operations (traffic + radio exchanges)
# ─────────────────────────────────────────────────────────────────────────────

# Relative hourly traffic profile (UTC hours 0..23): morning and evening banks.
HOURLY_PROFILE = np.array([0.15, 0.1, 0.1, 0.15, 0.3, 0.55, 0.8, 1.0, 0.95, 0.8, 0.7, 0.65,
                           0.6, 0.65, 0.7, 0.75, 0.85, 1.0, 0.95, 0.85, 0.7, 0.5, 0.35, 0.25])


@dataclass
class GroundTruth:
    """Parameters used to generate the synthetic month."""

    peak_rate_per_h: float = 14.0       # mean entries/h at the busiest hour
    exchanges_per_flight: float = 4.0   # mean exchanges per flight in an empty sector
    extra_per_concurrent: float = 0.5   # extra exchanges per other flight present at entry
    pilot_s: float = 2.8                # mean pilot transmission (s)
    ctl_s: float = 4.2                  # mean controller transmission (s)
    transit_min: float = 12.0           # time in the sector
    misclass_rate: float = 0.05         # CTL <-> PILOT label swaps
    noise_per_h: float = 20.0           # spurious NOISE segments per hour

    def radio_service(self, alpha: float, n_present: float = 0.0) -> float:
        """Expected D_alpha per flight when ``n_present`` other flights are in the sector.

        Service grows with concurrent traffic (coordination, sequencing and
        conflict-resolution exchanges), which is what makes the occupancy ->
        throughput relation saturate.
        """
        n_ex = self.exchanges_per_flight + self.extra_per_concurrent * n_present
        return n_ex * (self.ctl_s + alpha * self.pilot_s)


def simulate_operations(days: int = 28, start: str = "2024-02-01", truth: GroundTruth | None = None,
                        seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (predictions-like segments, traffic) for ``days`` of operations."""
    truth = truth or GroundTruth()
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)

    # 1. Traffic: non-homogeneous Poisson entries, hour by hour.
    entries = []
    for d in range(days):
        for h in range(24):
            n = rng.poisson(truth.peak_rate_per_h * HOURLY_PROFILE[h])
            base = d * 86400 + h * 3600
            entries.extend(base + rng.uniform(0, 3600, n))
    entries = np.sort(np.array(entries))

    # 2. Requested exchanges: first call a few minutes before entry, the rest
    #    spread over the transit; each is a pilot call then a controller reply.
    transit_s = truth.transit_min * 60
    # Flights already in the sector at each entry (entered within one transit).
    n_present = np.arange(len(entries)) - np.searchsorted(entries, entries - transit_s)
    calls = []
    for k, te in enumerate(entries):
        lam = truth.exchanges_per_flight + truth.extra_per_concurrent * n_present[k]
        n_ex = max(1, rng.poisson(lam))
        first = te - rng.uniform(60, 300)
        times = np.sort(np.concatenate([[first], te + rng.uniform(0, transit_s, n_ex - 1)]))
        for t in times:
            calls.append((t, k))
    calls.sort()

    # 3. Single half-duplex frequency: an exchange starts when the channel is free.
    segs, free_at = [], -np.inf
    for t, k in calls:
        s = max(t, free_at)
        d_pil = rng.gamma(4.0, truth.pilot_s / 4.0)
        d_ctl = rng.gamma(4.0, truth.ctl_s / 4.0)
        gap = rng.uniform(0.3, 1.0)
        segs.append((s, d_pil, "PILOT", k))
        segs.append((s + d_pil + gap, d_ctl, "CTL", k))
        free_at = s + d_pil + gap + d_ctl + rng.uniform(0.5, 2.0)

    df = pd.DataFrame(segs, columns=["t_s", "duration_s", "truth", "flight"])

    # 4. Channel noise segments (carrier, double transmissions, beeps).
    n_noise = rng.poisson(truth.noise_per_h * 24 * days)
    noise = pd.DataFrame({"t_s": rng.uniform(0, days * 86400, n_noise),
                          "duration_s": rng.uniform(0.5, 2.0, n_noise),
                          "truth": "NOISE", "flight": -1})
    df = pd.concat([df, noise], ignore_index=True).sort_values("t_s", ignore_index=True)

    # 5. Classifier behaviour: a few label swaps; lower confidence when wrong.
    pred = df["truth"].to_numpy().copy()
    swap = (rng.random(len(df)) < truth.misclass_rate) & (pred != "NOISE")
    pred[swap] = np.where(pred[swap] == "CTL", "PILOT", "CTL")
    conf = np.where(swap, rng.uniform(0.45, 0.85, len(df)), rng.beta(12, 1.5, len(df)))
    df["prediction"] = pred
    df["confidence"] = conf.round(3)

    # 6. Lay out as 30-minute recording files, like the TopSky export.
    df["t_start"] = t0 + pd.to_timedelta(df["t_s"], unit="s")
    file_start = df["t_start"].dt.floor("30min")
    df["source_file"] = file_start.dt.strftime("%Y_%-m_%-d_%-H_%-M_0") + "_ch17.wav"
    df["start_s"] = (df["t_start"] - file_start).dt.total_seconds().round(3)
    df["end_s"] = (df["start_s"] + df["duration_s"]).round(3)
    df["duration_s"] = df["duration_s"].round(3)
    df["segment_id"] = [f"seg{i:06d}" for i in range(len(df))]
    predictions = df[["segment_id", "source_file", "start_s", "end_s", "duration_s",
                      "prediction", "confidence", "t_start", "truth", "flight"]]

    t_entry = t0 + pd.to_timedelta(entries, unit="s")
    traffic = pd.DataFrame({
        "callsign": [f"SYN{i:04d}" for i in range(len(entries))],
        "t_entry": t_entry,
        "t_exit": t_entry + pd.to_timedelta(truth.transit_min, unit="min"),
    })
    return predictions, traffic


# ─────────────────────────────────────────────────────────────────────────────
# Audio
# ─────────────────────────────────────────────────────────────────────────────

def _voice(rng, dur, sr, f0_range, n_harm):
    """Harmonic 'vowel' stream with syllable-rate envelope and pitch drift."""
    t = np.arange(int(dur * sr)) / sr
    f0 = rng.uniform(*f0_range) * (1 + 0.04 * np.sin(2 * np.pi * rng.uniform(0.5, 2) * t))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    formants = rng.uniform([500, 1300, 2400], [800, 1900, 3000])
    sig = np.zeros_like(t)
    for h in range(1, n_harm + 1):
        fh = h * f0.mean()
        gain = sum(np.exp(-((fh - f) / 250.0) ** 2) for f in formants) + 0.05
        sig += gain / h**0.5 * np.sin(h * phase)
    env = np.clip(np.sin(2 * np.pi * rng.uniform(3, 5) * t + rng.uniform(0, 6)), 0, None) ** 0.6
    return sig * (0.3 + 0.7 * env)


def _bandpass(x, sr, lo, hi):
    return sosfilt(butter(4, [lo, hi], btype="band", fs=sr, output="sos"), x)


def synth_segment(kind: str, dur: float, sr: int, rng) -> np.ndarray:
    """One synthetic transmission of kind CTL, PILOT or NOISE."""
    if kind == "CTL":
        x = _voice(rng, dur, sr, (95, 140), 40)
        x += 0.01 * rng.standard_normal(len(x))
    elif kind == "PILOT":
        x = _bandpass(_voice(rng, dur, sr, (110, 210), 25), sr, 300, 3000)
        x = np.tanh(3 * x / (np.abs(x).max() + 1e-9))   # VHF clipping
        x += 0.12 * rng.standard_normal(len(x))
    else:
        t = np.arange(int(dur * sr)) / sr
        if rng.random() < 0.5:   # carrier / beep
            x = np.sin(2 * np.pi * rng.choice([1000, 1020, 2000]) * t)
        else:                    # broadband burst
            x = _bandpass(rng.standard_normal(len(t)), sr, 200, 6000)
    return (0.3 * x / (np.abs(x).max() + 1e-9)).astype(np.float32)


def make_audio_corpus(out_dir: str | Path, n_files: int = 30, file_s: float = 40.0, sr: int = 16000,
                      seed: int = 3) -> pd.DataFrame:
    """Write ``n_files`` WAV files; return the true (file, start, end, label) table."""
    import soundfile as sf

    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    truth = []
    for i in range(n_files):
        y = 0.002 * rng.standard_normal(int(file_s * sr)).astype(np.float32)
        t = rng.uniform(0.5, 2.0)
        while t < file_s - 4:
            kind = rng.choice(["CTL", "PILOT", "NOISE"], p=[0.42, 0.42, 0.16])
            dur = rng.uniform(1.0, 4.0) if kind != "NOISE" else rng.uniform(0.6, 1.5)
            a = int(t * sr)
            seg = synth_segment(kind, dur, sr, rng)
            y[a : a + len(seg)] += seg[: len(y) - a]
            truth.append({"source_file": str(out / f"rec_{i:03d}_ch17.wav"), "start_s": t,
                          "end_s": t + dur, "label": kind})
            t += dur + rng.uniform(0.8, 3.0)
        sf.write(str(out / f"rec_{i:03d}_ch17.wav"), y, sr)
    return pd.DataFrame(truth)


def label_by_overlap(segments: pd.DataFrame, truth: pd.DataFrame, min_overlap: float = 0.5) -> pd.DataFrame:
    """Annotate VAD segments with the true label they mostly overlap.

    Stands in for the manual listening step on synthetic data only.
    """
    out = segments.copy()
    labels = []
    for _, s in out.iterrows():
        cand = truth[truth["source_file"] == s["source_file"]]
        ov = (np.minimum(cand["end_s"], s["end_s"]) - np.maximum(cand["start_s"], s["start_s"])).clip(lower=0)
        dur = s["end_s"] - s["start_s"]
        j = ov.idxmax() if len(ov) else None
        labels.append(cand.loc[j, "label"] if j is not None and ov[j] >= min_overlap * dur else "")
    out["label"] = labels
    return out
