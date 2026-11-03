# Data

No operational data is versioned here. Put real files in `data/real/` (ignored by git).

## Recordings

Audio files (`.wav`, `.mp3`, `.flac`) of the executive radio position. Absolute times are read
from TopSky export names: `YYYY_M_D_H_M_S_chNN`, for example `2024_2_1_10_30_0_ch17.wav`
(UTC, start of the file).

## Predictions (`atsvc predict` output)

| column | content |
|---|---|
| `segment_id` | unique id |
| `source_file` | recording path |
| `start_s`, `end_s`, `duration_s` | segment position in the file (s) |
| `prediction` | `CTL`, `PILOT` or `NOISE` (`PILOTE`, `BRUIT` also accepted) |
| `confidence` | highest class probability, 0–1 |
| `t_start` | absolute start time; added by `atsvc.io.add_absolute_times` |

## Traffic

Either a CSV with `t_entry` (and optionally `t_exit`) timestamps, or a flight statistics export
read by `atsvc.io.load_traffic`: one row per flight, a date column and entry / exit time columns
(names configurable; defaults `Date du vol`, `Hreentrée`, `Hresortie`). The header row is found
automatically. Use `shift_to_utc_h` if traffic times are local while recordings are UTC.

## Synthetic data

`atsvc simulate --out-dir data/synthetic` writes `predictions.csv` and `traffic.csv` for a synthetic
month (see `atsvc.synthetic`). Synthetic audio for notebook 1 is regenerated in
`data/synthetic_audio/` on each run and is not versioned.
