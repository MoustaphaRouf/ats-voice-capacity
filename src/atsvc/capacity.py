r"""Sector capacity from controller radio occupancy (Audio-DORATASK x Radio-Traffic).

Two independent estimates of capacity are confronted:

1. **Analytical (Audio-DORATASK).** If a fraction :math:`\varphi_c` of the
   controller's working time is spent on voice exchanges and each movement
   requires :math:`d_{radio}` seconds of radio service, then

   .. math:: C_{DORA}(\varphi_c) = \varphi_c \cdot 3600 / d_{radio}

   a straight line through the origin.

2. **Empirical (Radio-Traffic).** Sliding windows of the recordings give the
   observed radio occupancy :math:`X` and the traffic entry rate :math:`Y_2`.
   A saturating curve :math:`C_{emp}(X)` (Michaelis-Menten by default) is
   fitted on real windows augmented by superposing busy windows.

:math:`\varphi_c` is not assumed: the intersection of the line and the curve
gives the voice fraction :math:`\varphi_c^*` consistent with both approaches,
and the capacity :math:`C^*` at that point. This matters for sectors that have
never been saturated, where capacity cannot be read off the traffic itself.

Notation follows the paper: *ATS Capacity Estimation From Control Center Audio
Recordings* (SSRN, 2026).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq, curve_fit

from .classifier import normalise_label

ESTIMATORS = {"mean": "Mean", "p50": "Median", "p70": "P70", "p85": "P85"}


# ─────────────────────────────────────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────────────────────────────────────

def reliable_speech(predictions: pd.DataFrame, min_confidence: float = 0.70) -> pd.DataFrame:
    """Keep confidently classified CTL / PILOT segments with absolute times.

    ``predictions`` needs ``t_start`` (timestamp) and ``duration_s``, plus
    ``prediction`` and ``confidence`` from the classifier.
    """
    df = predictions.copy()
    df["label"] = df["prediction"].map(normalise_label)
    df = df[df["label"].isin(["CTL", "PILOT"]) & (df["confidence"] >= min_confidence)].copy()
    df["t_start"] = pd.to_datetime(df["t_start"])
    df["t_end"] = df["t_start"] + pd.to_timedelta(df["duration_s"], unit="s")
    return df


def hourly_table(speech: pd.DataFrame, entries: pd.Series, alphas=(0.2, 0.7)) -> pd.DataFrame:
    """Hourly controller / pilot speech durations and TMA entries N(h).

    ``D_alpha(h) = D_CTL(h) + alpha * D_PIL(h)``: pilot speech occupies the
    frequency and the controller's attention, with a weight alpha that depends
    on the operational environment and must be set per sector.
    """
    entries = pd.to_datetime(pd.Series(entries)).dropna()
    s = speech.assign(hour=speech["t_start"].dt.floor("h"))
    load = (s.pivot_table(index="hour", columns="label", values="duration_s", aggfunc="sum")
              .reindex(columns=["CTL", "PILOT"]).fillna(0.0)
              .rename(columns={"CTL": "D_ctl_s", "PILOT": "D_pil_s"}))
    start = min(entries.min(), s["t_start"].min()).floor("h")
    end = max(entries.max(), s["t_end"].max()).ceil("h")
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    n_h = entries.dt.floor("h").value_counts().reindex(hours, fill_value=0)
    out = pd.DataFrame({"hour": hours, "N_h": n_h.to_numpy()})
    out = out.merge(load, left_on="hour", right_index=True, how="left").fillna({"D_ctl_s": 0.0, "D_pil_s": 0.0})
    for a in alphas:
        out[f"D_alpha_{a}"] = out["D_ctl_s"] + a * out["D_pil_s"]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Analytical line
# ─────────────────────────────────────────────────────────────────────────────

def peak_hours(hourly: pd.DataFrame, alpha: float, percentile: float = 0.75,
               min_audio_s: float = 30.0, min_movements: int = 1) -> pd.DataFrame:
    """Busy hours: traffic present, enough audio, top quartile of N(h)."""
    active = hourly[(hourly["N_h"] >= min_movements) & (hourly[f"D_alpha_{alpha}"] >= min_audio_s)]
    return active[active["N_h"] >= active["N_h"].quantile(percentile)]


def estimate_d_radio(ratios: pd.Series, estimator: str) -> float:
    """Radio service per movement from the hourly ratios D_alpha(h) / N(h).

    mean = typical service, sensitive to atypical hours; p50 = robust typical
    service; p70 = moderately unfavourable (prudent sizing); p85 = strongly
    unfavourable (safety bound). Higher estimators give lower capacities.
    """
    r = pd.Series(ratios, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if estimator == "mean":
        return float(r.mean())
    return float(r.quantile({"p50": 0.50, "p70": 0.70, "p85": 0.85}[estimator]))


def doratask_lines(hourly: pd.DataFrame, alphas=(0.2, 0.7), estimators=tuple(ESTIMATORS),
                   percentile: float = 0.75) -> pd.DataFrame:
    """d_radio and slope 3600 / d_radio for each (alpha, estimator)."""
    rows = []
    for a in alphas:
        peak = peak_hours(hourly, a, percentile)
        ratios = peak[f"D_alpha_{a}"] / peak["N_h"]
        for est in estimators:
            d = estimate_d_radio(ratios, est)
            rows.append({"alpha": a, "estimator": est, "label": ESTIMATORS[est],
                         "n_peak_hours": len(peak), "d_radio_s": d, "slope": 3600.0 / d})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Empirical curve
# ─────────────────────────────────────────────────────────────────────────────

class Raster:
    """Per-second series: controller speech, pilot speech, TMA entries."""

    def __init__(self, speech: pd.DataFrame, entries: pd.Series):
        entries = pd.to_datetime(pd.Series(entries)).dropna()
        self.t0 = speech["t_start"].min().floor("min")
        t1 = speech["t_end"].max().ceil("min")
        self.n = int((t1 - self.t0).total_seconds())
        self.ctl = np.zeros(self.n, dtype=np.float32)
        self.pil = np.zeros(self.n, dtype=np.float32)
        self.ent = np.zeros(self.n, dtype=np.float32)
        for label, arr in (("CTL", self.ctl), ("PILOT", self.pil)):
            sub = speech[speech["label"] == label]
            i0 = ((sub["t_start"] - self.t0).dt.total_seconds()).astype(int).to_numpy()
            i1 = ((sub["t_end"] - self.t0).dt.total_seconds()).astype(int).to_numpy()
            for a, b in zip(np.clip(i0, 0, self.n), np.clip(i1, 0, self.n)):
                arr[a:b] = 1.0
        idx = ((entries - self.t0).dt.total_seconds()).astype(int).to_numpy()
        idx = idx[(idx >= 0) & (idx < self.n)]
        np.add.at(self.ent, idx, 1.0)
        # Cumulative sums make every window sum O(1).
        self._c = {k: np.concatenate([[0.0], np.cumsum(v, dtype=np.float64)])
                   for k, v in (("ctl", self.ctl), ("pil", self.pil), ("ent", self.ent))}

    def _sum(self, key, a, b):
        c = self._c[key]
        return c[b] - c[a]

    def windows(self, win_min: int, step_min: int = 5, before_min: int = 10, after_min: int = 5) -> pd.DataFrame:
        """Real sliding windows: speech seconds and entry rate Y2 (flights/h).

        Y2 counts entries over the asymmetric window [t - before, t + W + after]
        because pilots call before entering and some enter after the call.
        """
        w, st, bf, af = win_min * 60, step_min * 60, before_min * 60, after_min * 60
        starts = np.arange(bf, self.n - w - af + 1, st)
        if len(starts) == 0:
            return pd.DataFrame(columns=["start", "ctl_s", "pil_s", "Y2"])
        return pd.DataFrame({
            "start": starts,
            "ctl_s": self._sum("ctl", starts, starts + w),
            "pil_s": self._sum("pil", starts, starts + w),
            "Y2": self._sum("ent", starts - bf, starts + w + af) / ((w + bf + af) / 3600.0),
        })


def occupancy(win: pd.DataFrame, alpha: float, win_min: int) -> np.ndarray:
    """Radio occupancy X = (D_CTL + alpha D_PIL) / W."""
    return ((win["ctl_s"] + alpha * win["pil_s"]) / (win_min * 60)).to_numpy()


def superpose(win: pd.DataFrame, alpha: float, win_min: int, n_draws: int = 8000, n_max: int = 4,
              q_min: float = 0.70, seed: int = 42) -> pd.DataFrame:
    """Synthetic heavier-traffic windows by superposing busy real windows.

    Observed traffic rarely saturates the sector, so the observed X domain is
    short. Each draw stacks 1..n_max distinct windows from the top (1-q_min)
    of X; X and Y2 of the stack are recomputed from the stacked series, so the
    physical X -> Y2 relation is preserved. Because both quantities are sums,
    the stack's X and Y2 equal the sums of the stacked windows' X and Y2, which
    is what is computed here (identical to the original second-by-second
    accumulation, several hundred times faster). Stacks with X > 1 are dropped.
    Assumes the superposed traffics are independent.
    """
    x = occupancy(win, alpha, win_min)
    y = win["Y2"].to_numpy()
    top = np.flatnonzero(x >= np.quantile(x, q_min))
    if len(top) < n_max + 1:
        return pd.DataFrame(columns=["X", "Y2", "level"])
    rng = np.random.default_rng(seed)
    picks = np.array([rng.choice(top, size=n_max, replace=False) for _ in range(n_draws)])
    cx, cy = np.cumsum(x[picks], axis=1), np.cumsum(y[picks], axis=1)
    level = np.broadcast_to(np.arange(1, n_max + 1), cx.shape)
    keep = cx <= 1.0
    return pd.DataFrame({"X": cx[keep], "Y2": cy[keep], "level": level[keep]})


def michaelis_menten(x, c_max, k):
    """Saturating through the origin: C_max x / (k + x)."""
    return c_max * np.asarray(x, float) / (k + np.asarray(x, float) + 1e-12)


def gompertz_origin(x, a, b, c):
    """Gompertz shifted through the origin (sensitivity model of the paper)."""
    x = np.asarray(x, float)
    return a * (np.exp(-np.exp(b - c * x)) - np.exp(-np.exp(b)))


MODELS = {
    "michaelis-menten": (michaelis_menten, lambda x, y: [max(y.max(), 1.0), max(np.median(x), 1e-3)],
                         ([0.1, 1e-6], [500.0, 10.0])),
    "gompertz": (gompertz_origin, lambda x, y: [max(y.max(), 1.0), 1.0, 8.0],
                 ([0.1, -5.0, 0.1], [500.0, 10.0, 200.0])),
}


def fit_curve(x, y, model: str = "michaelis-menten"):
    """Least-squares fit; returns (params, callable) or (None, None)."""
    f, p0, bounds = MODELS[model]
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 10:
        return None, None
    try:
        popt, _ = curve_fit(f, x, y, p0=p0(x, y), bounds=bounds, maxfev=20000)
    except (RuntimeError, ValueError):
        return None, None
    return popt, (lambda t, p=popt: f(t, *p))


def intersection(curve, slope: float, phi_range=(0.02, 0.70), phi_min: float = 0.03):
    """First non-trivial crossing of C = slope * phi with the empirical curve."""
    def diff(p):
        return slope * p - float(curve(p))
    grid = np.linspace(*phi_range, 900)
    vals = np.array([diff(p) for p in grid])
    for k in np.flatnonzero(np.diff(np.sign(vals)) != 0):
        root = brentq(diff, grid[k], grid[k + 1])
        if root >= phi_min:
            return float(root), float(slope * root)
    return np.nan, np.nan


def implied_service(curve, x) -> np.ndarray:
    r"""Per-flight radio service along the empirical curve: s(X) = 3600 X / C_emp(X).

    Identification diagnostic. At an intersection, phi* 3600 / d_radio =
    C_emp(phi*), i.e. s(phi*) = d_radio: C* is the throughput at which the
    fitted per-flight service reaches the chosen d_radio estimator. The
    flatter s(X) is, the more C* moves with the estimator (and the further it
    may sit beyond the real X range, where the curve is extrapolated).
    """
    x = np.asarray(x, float)
    return 3600.0 * x / np.asarray(curve(x), float)


def r2_mae(y, y_hat):
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - y_hat) ** 2) / ss_tot if ss_tot > 0 else np.nan
    return float(r2), float(np.mean(np.abs(y - y_hat)))


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CapacityResult:
    hourly: pd.DataFrame
    lines: pd.DataFrame
    fits: pd.DataFrame
    intersections: pd.DataFrame
    selected: dict = field(default_factory=dict)   # alpha -> window (min)
    curves: dict = field(default_factory=dict)     # (alpha, window) -> callable
    real: dict = field(default_factory=dict)       # (alpha, window) -> (X, Y2) real
    synthetic: dict = field(default_factory=dict)  # (alpha, window) -> DataFrame

    def summary(self) -> pd.DataFrame:
        """C* range per alpha for the selected window."""
        rows = []
        for a, w in self.selected.items():
            sub = self.intersections[(self.intersections["alpha"] == a) & (self.intersections["window_min"] == w)]
            for _, r in sub.iterrows():
                rows.append({"alpha": a, "window_min": w, "estimator": r["label"],
                             "d_radio_s": r["d_radio_s"], "phi_c_star": r["phi_c_star"],
                             "C_star": r["C_star"], "extrapolated": r["extrapolated"]})
        return pd.DataFrame(rows)


def estimate_capacity(predictions: pd.DataFrame, entries: pd.Series, alphas=(0.2, 0.7),
                      windows=(60, 90, 120, 180), model: str = "michaelis-menten",
                      min_confidence: float = 0.70, mae_max: float = 2.0, n_draws: int = 8000,
                      seed: int = 42) -> CapacityResult:
    """Full pipeline from classified segments and TMA entry times to C*.

    Selection rule (per alpha): among windows whose curve crosses at least one
    DORATASK line and whose MAE on real windows is below ``mae_max``, keep the
    one with the best R^2 on real windows (augmented windows are used for
    fitting only, never for scoring).
    """
    speech = reliable_speech(predictions, min_confidence)
    hourly = hourly_table(speech, entries, alphas)
    lines = doratask_lines(hourly, alphas)
    raster = Raster(speech, entries)
    res = CapacityResult(hourly, lines, pd.DataFrame(), pd.DataFrame())
    fit_rows, inter_rows = [], []
    for w in windows:
        win = raster.windows(w)
        for a in alphas:
            real = pd.DataFrame({"X": occupancy(win, a, w), "Y2": win["Y2"].to_numpy()})
            real = real[real["Y2"] > 0]
            synth = superpose(win, a, w, n_draws=n_draws, seed=seed)
            fit_df = pd.concat([real.assign(level=0), synth], ignore_index=True)
            params, curve = fit_curve(fit_df["X"], fit_df["Y2"], model)
            if curve is None:
                fit_rows.append({"alpha": a, "window_min": w, "fitted": False})
                continue
            r2, mae = r2_mae(real["Y2"], curve(real["X"]))
            n_cross = 0
            for _, ln in lines[lines["alpha"] == a].iterrows():
                phi, c = intersection(curve, ln["slope"])
                n_cross += int(np.isfinite(phi))
                inter_rows.append({"alpha": a, "window_min": w, "estimator": ln["estimator"],
                                   "label": ln["label"], "d_radio_s": ln["d_radio_s"],
                                   "phi_c_star": phi, "C_star": c,
                                   # beyond the observed X range the curve is extrapolated
                                   "extrapolated": bool(np.isfinite(phi) and phi > real["X"].max())})
            fit_rows.append({"alpha": a, "window_min": w, "fitted": True, "params": np.round(params, 4).tolist(),
                             "r2_real": r2, "mae_real": mae, "n_real": len(real), "n_synthetic": len(synth),
                             "n_crossings": n_cross, "X_max_real": float(real["X"].max())})
            res.curves[(a, w)] = curve
            res.real[(a, w)] = real
            res.synthetic[(a, w)] = synth
    res.fits = pd.DataFrame(fit_rows)
    res.intersections = pd.DataFrame(inter_rows)
    for a in alphas:
        cand = res.fits[(res.fits["alpha"] == a) & res.fits["fitted"]
                        & (res.fits["n_crossings"] >= 1) & (res.fits["mae_real"] < mae_max)]
        if len(cand):
            res.selected[a] = int(cand.sort_values("r2_real", ascending=False).iloc[0]["window_min"])
    return res
