"""CTL / PILOT / NOISE classifier for speech segments.

Workflow (mirrors the original three-step notebook, Steps A-C):

1. ``build_annotation_set``: segment a sample of recordings, export each
   segment as a short WAV to listen to, and write a CSV with an empty
   ``label`` column to fill in by hand.
2. ``train``: fit a Random Forest on the labelled CSV and report
   cross-validated performance.
3. ``predict``: segment and label every recording of a folder.

Changes from the original code
------------------------------
* Scaling now happens inside a ``Pipeline``, so cross-validation no longer
  leaks the scaler fitted on the full corpus into the validation folds.
* The reported confusion matrix comes from out-of-fold predictions over the
  whole corpus (``cross_val_predict``) instead of a separate 20 % split
  evaluated after cross-validation, so every labelled segment is used once
  for evaluation and the two reported numbers are consistent.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .audio import FEATURE_COLUMNS, extract_features, load_audio, segment_vad

LABELS = ("CTL", "PILOT", "NOISE")
# Labels used in the original French notebooks, mapped to the English ones.
LABEL_ALIASES = {"PILOTE": "PILOT", "BRUIT": "NOISE", "ATCO": "CTL",
                 "DOUBLE": "NOISE", "PORTEUSE": "NOISE", "BIP": "NOISE"}
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac")


def normalise_label(label: str) -> str:
    lab = str(label).strip().upper()
    return LABEL_ALIASES.get(lab, lab)


def list_recordings(folder: str | Path) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS)


def segment_table(paths: list[Path], sr: int = 16000, export_dir: str | Path | None = None) -> pd.DataFrame:
    """Segment recordings and describe each segment; one row per segment."""
    rows = []
    if export_dir is not None:
        import soundfile as sf

        Path(export_dir).mkdir(parents=True, exist_ok=True)
    for path in paths:
        y, sr_ = load_audio(path, sr=sr)
        for i, seg in enumerate(segment_vad(y, sr_)):
            feats = extract_features(seg.audio, seg.sr)
            if feats is None:
                continue
            seg_id = f"{path.stem}_seg{i:02d}"
            if export_dir is not None:
                sf.write(str(Path(export_dir) / f"{seg_id}.wav"), seg.audio, seg.sr)
            rows.append({"segment_id": seg_id, "source_file": str(path),
                         "start_s": seg.start_s, "end_s": seg.end_s, **feats})
    return pd.DataFrame(rows)


def build_annotation_set(folder: str | Path, n_files: int = 250, out_csv: str | Path = "to_annotate.csv",
                         export_dir: str | Path = "segments_to_listen", seed: int = 42) -> pd.DataFrame:
    """Step A: sample recordings, export segments, write the CSV to label."""
    paths = list_recordings(folder)
    if len(paths) > n_files:
        rng = np.random.default_rng(seed)
        paths = [paths[i] for i in sorted(rng.choice(len(paths), n_files, replace=False))]
    df = segment_table(paths, export_dir=export_dir)
    df.insert(4, "label", "")
    df.to_csv(out_csv, index=False)
    return df


@dataclass
class TrainingReport:
    n_examples: int
    class_counts: dict
    cv_accuracy: float
    report: str
    confusion: pd.DataFrame
    top_features: pd.Series


def make_pipeline(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=seed)),
    ])


def train(annotations: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> tuple[Pipeline, TrainingReport]:
    """Step B: fit the classifier and evaluate it by stratified k-fold CV."""
    df = annotations[annotations["label"].notna() & (annotations["label"].astype(str).str.strip() != "")].copy()
    df["label"] = df["label"].map(normalise_label)
    X = df[FEATURE_COLUMNS].fillna(0.0).to_numpy()
    y = df["label"].to_numpy()

    min_class = df["label"].value_counts().min()
    folds = max(2, min(n_folds, int(min_class)))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    y_oof = cross_val_predict(make_pipeline(seed), X, y, cv=cv)

    labels = [l for l in LABELS if l in set(y)]
    model = make_pipeline(seed).fit(X, y)
    importances = pd.Series(model.named_steps["rf"].feature_importances_, index=FEATURE_COLUMNS)
    report = TrainingReport(
        n_examples=len(df),
        class_counts=df["label"].value_counts().to_dict(),
        cv_accuracy=float(accuracy_score(y, y_oof)),
        report=classification_report(y, y_oof, labels=labels, zero_division=0),
        confusion=pd.DataFrame(confusion_matrix(y, y_oof, labels=labels), index=labels, columns=labels),
        top_features=importances.sort_values(ascending=False).head(10),
    )
    return model, report


def save_model(model: Pipeline, path: str | Path) -> None:
    with open(path, "wb") as f:
        pickle.dump({"model": model, "columns": FEATURE_COLUMNS}, f)


def load_model(path: str | Path) -> Pipeline:
    """Load a model saved by :func:`save_model`. Only load files you trust."""
    with open(path, "rb") as f:
        return pickle.load(f)["model"]


def predict_table(model: Pipeline, segments: pd.DataFrame) -> pd.DataFrame:
    """Label an already-segmented table (output of :func:`segment_table`)."""
    X = segments[FEATURE_COLUMNS].fillna(0.0).to_numpy()
    proba = model.predict_proba(X)
    out = segments[["segment_id", "source_file", "start_s", "end_s", "duration_s"]].copy()
    out["prediction"] = model.classes_[proba.argmax(axis=1)]
    out["confidence"] = proba.max(axis=1).round(3)
    return out


def predict(model: Pipeline, folder: str | Path) -> pd.DataFrame:
    """Step C: segment and label every recording in ``folder``."""
    return predict_table(model, segment_table(list_recordings(folder)))
