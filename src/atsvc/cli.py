"""Command line: atsvc {annotate,train,predict,capacity,simulate}."""
from __future__ import annotations

import argparse

import pandas as pd


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="atsvc", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("annotate", help="segment a sample of recordings for manual labelling")
    a.add_argument("folder"); a.add_argument("--n-files", type=int, default=250)
    a.add_argument("--out", default="to_annotate.csv"); a.add_argument("--export-dir", default="segments_to_listen")

    t = sub.add_parser("train", help="train the CTL/PILOT/NOISE classifier")
    t.add_argument("annotations"); t.add_argument("--model", default="acoustic_model.pkl")

    pr = sub.add_parser("predict", help="label every recording of a folder")
    pr.add_argument("folder"); pr.add_argument("--model", default="acoustic_model.pkl")
    pr.add_argument("--out", default="predictions.csv")

    c = sub.add_parser("capacity", help="estimate C* from predictions and traffic")
    c.add_argument("predictions"); c.add_argument("traffic")
    c.add_argument("--start"); c.add_argument("--end")
    c.add_argument("--alphas", type=float, nargs="+", default=[0.2, 0.7])
    c.add_argument("--model", default="michaelis-menten", choices=["michaelis-menten", "gompertz"])
    c.add_argument("--out", default="capacity_summary.csv")

    s = sub.add_parser("simulate", help="write a synthetic month (predictions + traffic)")
    s.add_argument("--days", type=int, default=28); s.add_argument("--out-dir", default="data/synthetic")

    args = p.parse_args(argv)

    if args.cmd == "annotate":
        from .classifier import build_annotation_set
        df = build_annotation_set(args.folder, args.n_files, args.out, args.export_dir)
        print(f"{len(df)} segments -> {args.out}; listen in {args.export_dir}/ and fill 'label' (CTL/PILOT/NOISE).")
    elif args.cmd == "train":
        from .classifier import save_model, train
        model, rep = train(pd.read_csv(args.annotations))
        print(f"{rep.n_examples} examples {rep.class_counts}\nCV accuracy {rep.cv_accuracy:.1%}\n{rep.report}\n{rep.confusion}")
        save_model(model, args.model)
    elif args.cmd == "predict":
        from .classifier import load_model, predict
        df = predict(load_model(args.model), args.folder)
        df.to_csv(args.out, index=False)
        print(f"{len(df)} segments -> {args.out}\n{df['prediction'].value_counts()}")
    elif args.cmd == "capacity":
        from .capacity import estimate_capacity
        from .io import add_absolute_times, load_traffic
        pred = pd.read_csv(args.predictions)
        if "t_start" not in pred.columns:
            pred = add_absolute_times(pred)
        traffic = (pd.read_csv(args.traffic, parse_dates=["t_entry", "t_exit"])
                   if args.traffic.endswith(".csv") and "t_entry" in pd.read_csv(args.traffic, nrows=0).columns
                   else load_traffic(args.traffic, start=args.start, end=args.end))
        res = estimate_capacity(pred, traffic["t_entry"], alphas=tuple(args.alphas), model=args.model)
        print(res.lines.round(2).to_string(index=False))
        print(res.fits.drop(columns="params", errors="ignore").round(3).to_string(index=False))
        summary = res.summary()
        print(summary.round(3).to_string(index=False))
        summary.to_csv(args.out, index=False)
    elif args.cmd == "simulate":
        from pathlib import Path
        from .synthetic import simulate_operations
        out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
        pred, traffic = simulate_operations(days=args.days)
        pred.to_csv(out / "predictions.csv", index=False)
        traffic.to_csv(out / "traffic.csv", index=False)
        print(f"{len(pred)} segments, {len(traffic)} flights -> {out}/")


if __name__ == "__main__":
    main()
