"""Loading real inputs: TopSky recording timestamps and traffic statistics."""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime, time as dt_time
from pathlib import Path

import pandas as pd

# TopSky export file names: YYYY_M_D_H_M_S_chNN (e.g. 2024_2_1_10_30_0_ch17.wav)
_TOPSKY_TS = re.compile(r"(\d{4})_(\d{1,2})_(\d{1,2})_(\d{1,2})_(\d{1,2})_(\d{1,2})_ch\d+")


def recording_start(path: str | Path) -> pd.Timestamp:
    """Start time of a TopSky recording, read from its file name."""
    m = _TOPSKY_TS.search(Path(str(path)).name)
    if not m:
        return pd.NaT
    y, mo, d, h, mi, s = map(int, m.groups())
    return pd.Timestamp(y, mo, d, h, mi, s)


def add_absolute_times(predictions: pd.DataFrame) -> pd.DataFrame:
    """Add ``t_start`` = file start + segment offset to classifier output."""
    df = predictions.copy()
    df["t_start"] = df["source_file"].map(recording_start) + pd.to_timedelta(df["start_s"], unit="s")
    return df.dropna(subset=["t_start"])


def _norm(s) -> str:
    s = "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s.strip()).lower()


def _find_column(df: pd.DataFrame, name: str) -> str:
    target = _norm(name)
    cols = {_norm(c): c for c in df.columns}
    if target in cols:
        return cols[target]
    for k, v in cols.items():
        if target in k or k in target:
            return v
    raise KeyError(f"Column not found: {name!r}. Available: {list(df.columns)}")


def _combine(date, clock, shift_h: float = 0.0) -> pd.Timestamp:
    """Date + heterogeneous Excel time (time, datetime, day fraction or text)."""
    if pd.isna(date) or pd.isna(clock):
        return pd.NaT
    d = pd.Timestamp(date).normalize()
    if isinstance(clock, dt_time):
        ts = d + pd.Timedelta(hours=clock.hour, minutes=clock.minute, seconds=clock.second)
    elif isinstance(clock, (pd.Timestamp, datetime)):
        t = pd.Timestamp(clock)
        ts = d + pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
    elif isinstance(clock, (int, float)) and not isinstance(clock, bool):
        ts = d + pd.Timedelta(seconds=round(float(clock) % 1.0 * 86400))
    else:
        t = pd.to_datetime(str(clock).strip(), errors="coerce")
        if pd.isna(t):
            return pd.NaT
        ts = d + pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
    return ts + pd.Timedelta(hours=shift_h)


def load_traffic(path: str | Path, date_col: str = "Date du vol", entry_col: str = "Hreentrée",
                 exit_col: str = "Hresortie", start: str | None = None, end: str | None = None,
                 shift_to_utc_h: float = 0.0, nominal_transit_min: float = 12.0) -> pd.DataFrame:
    """Read a traffic statistics file (xls/xlsx/csv) into t_entry / t_exit.

    The header row is located automatically (exports often carry title rows).
    Default column names are those of the flight statistics export used in the study;
    pass your own for other formats. Times are shifted by ``shift_to_utc_h``
    to match the UTC recording timestamps; a missing bound is filled with a
    nominal transit, and exits before entries are moved to the next day.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        engine = "xlrd" if path.suffix.lower() == ".xls" else None
        raw = pd.read_excel(path, header=None, engine=engine)
        header = 0
        for i in range(min(30, len(raw))):
            vals = {_norm(v) for v in raw.iloc[i].tolist()}
            if _norm(date_col) in vals and (_norm(entry_col) in vals or _norm(exit_col) in vals):
                header = i
                break
        df = pd.read_excel(path, header=header, engine=engine)
    df = df.dropna(how="all")
    c_date, c_in, c_out = (_find_column(df, c) for c in (date_col, entry_col, exit_col))
    df[c_date] = pd.to_datetime(df[c_date], errors="coerce")
    if start:
        df = df[df[c_date] >= pd.Timestamp(start)]
    if end:
        df = df[df[c_date] <= pd.Timestamp(end)]
    out = pd.DataFrame({
        "t_entry": [_combine(d, t, shift_to_utc_h) for d, t in zip(df[c_date], df[c_in])],
        "t_exit": [_combine(d, t, shift_to_utc_h) for d, t in zip(df[c_date], df[c_out])],
    })
    transit = pd.Timedelta(minutes=nominal_transit_min)
    over = out["t_exit"] < out["t_entry"]
    out.loc[over, "t_exit"] += pd.Timedelta(days=1)
    out["t_exit"] = out["t_exit"].fillna(out["t_entry"] + transit)
    out["t_entry"] = out["t_entry"].fillna(out["t_exit"] - transit)
    return out.dropna().reset_index(drop=True)
