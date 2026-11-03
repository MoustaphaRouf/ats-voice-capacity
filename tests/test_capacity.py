import numpy as np
import pandas as pd
import pytest

from atsvc import capacity
from atsvc.synthetic import GroundTruth, simulate_operations


@pytest.fixture(scope="module")
def month():
    pred, traffic = simulate_operations(days=28, seed=7)
    return pred, traffic


@pytest.fixture(scope="module")
def result(month):
    pred, traffic = month
    return capacity.estimate_capacity(pred, traffic["t_entry"], n_draws=3000)


def test_superposition_equals_second_by_second_accumulation(month):
    pred, traffic = month
    speech = capacity.reliable_speech(pred)
    raster = capacity.Raster(speech, traffic["t_entry"])
    win = raster.windows(60)
    fast = capacity.superpose(win, 0.7, 60, n_draws=50, seed=1)
    # Brute force: stack the same draws second by second, as in the original notebook.
    x = capacity.occupancy(win, 0.7, 60)
    top = np.flatnonzero(x >= np.quantile(x, 0.70))
    rng = np.random.default_rng(1)
    picks = [rng.choice(top, size=4, replace=False) for _ in range(50)]
    w, bf, af = 3600, 600, 300
    rows = []
    for idx in picks:
        acc = np.zeros(w); acc_p = np.zeros(w); acc_e = np.zeros(w + bf + af)
        for lvl, i in enumerate(idx, 1):
            s = int(win["start"].iloc[i])
            acc += raster.ctl[s:s + w]; acc_p += raster.pil[s:s + w]
            acc_e += raster.ent[s - bf:s + w + af]
            X = (acc.sum() + 0.7 * acc_p.sum()) / w
            if X <= 1.0:
                rows.append((X, acc_e.sum() / ((w + bf + af) / 3600), lvl))
    brute = pd.DataFrame(rows, columns=["X", "Y2", "level"])
    np.testing.assert_allclose(fast[["X", "Y2"]].to_numpy(), brute[["X", "Y2"]].to_numpy(), rtol=1e-9)


def test_windows_count_entries(month):
    pred, traffic = month
    speech = capacity.reliable_speech(pred)
    raster = capacity.Raster(speech, traffic["t_entry"])
    win = raster.windows(60)
    s = int(win["start"].iloc[100])
    t_a = raster.t0 + pd.Timedelta(seconds=s - 600)
    t_b = raster.t0 + pd.Timedelta(seconds=s + 3600 + 300)
    n = ((traffic["t_entry"] >= t_a) & (traffic["t_entry"] < t_b)).sum()
    assert win["Y2"].iloc[100] == pytest.approx(n / 1.25)


def test_michaelis_menten_fit_recovers_parameters():
    x = np.linspace(0.01, 0.8, 200)
    params, f = capacity.fit_curve(x, capacity.michaelis_menten(x, 40.0, 0.3))
    assert params == pytest.approx([40.0, 0.3], rel=1e-4)


def test_d_radio_close_to_ground_truth(month, result):
    pred, traffic = month
    for alpha in (0.2, 0.7):
        peak = capacity.peak_hours(result.hourly, alpha)
        hours = set(peak["hour"])
        flights = traffic.assign(hour=traffic["t_entry"].dt.floor("h"))
        ids = set(np.flatnonzero(flights["hour"].isin(hours)))
        speech = pred[pred["flight"].isin(ids)]
        per_flight = (speech.pivot_table(index="flight", columns="truth", values="duration_s", aggfunc="sum")
                      .reindex(columns=["CTL", "PILOT"]).fillna(0))
        true_d = float((per_flight["CTL"] + alpha * per_flight["PILOT"]).mean())
        est = result.lines.query("alpha == @alpha and estimator == 'mean'")["d_radio_s"].item()
        assert abs(est - true_d) / true_d < 0.15


def test_intersection_identity(result):
    # At every crossing, the per-flight service implied by the curve equals d_radio.
    found = result.intersections.dropna(subset=["phi_c_star"])
    assert len(found) > 0
    for _, r in found.iterrows():
        curve = result.curves[(r["alpha"], r["window_min"])]
        s = capacity.implied_service(curve, r["phi_c_star"])
        assert float(s) == pytest.approx(r["d_radio_s"], rel=1e-6)


def test_selection_scores_on_real_windows_only(result):
    assert result.selected, "no admissible model"
    for a, w in result.selected.items():
        row = result.fits.query("alpha == @a and window_min == @w").iloc[0]
        real = result.real[(a, w)]
        r2, _ = capacity.r2_mae(real["Y2"], result.curves[(a, w)](real["X"]))
        assert r2 == pytest.approx(row["r2_real"])
