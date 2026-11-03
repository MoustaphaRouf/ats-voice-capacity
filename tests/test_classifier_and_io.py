import pandas as pd
import pytest

pytest.importorskip("librosa")
from atsvc import classifier, synthetic
from atsvc.io import add_absolute_times, recording_start


def test_recording_start_from_topsky_name():
    assert recording_start("x/2024_2_1_10_30_0_ch17.wav") == pd.Timestamp("2024-02-01 10:30:00")
    assert pd.isna(recording_start("notes.wav"))


def test_absolute_times():
    df = pd.DataFrame({"source_file": ["2024_2_1_10_30_0_ch17.wav"], "start_s": [12.5]})
    assert add_absolute_times(df)["t_start"].iloc[0] == pd.Timestamp("2024-02-01 10:30:12.5")


def test_classifier_end_to_end_on_synthetic_audio(tmp_path):
    truth = synthetic.make_audio_corpus(tmp_path, n_files=8, file_s=30.0, seed=1)
    segs = classifier.segment_table(classifier.list_recordings(tmp_path))
    labelled = synthetic.label_by_overlap(segs, truth)
    model, report = classifier.train(labelled)
    assert report.cv_accuracy > 0.8          # mechanics check on easy synthetic audio
    pred = classifier.predict_table(model, segs)
    assert set(pred["prediction"]) <= {"CTL", "PILOT", "NOISE"}
    assert pred["confidence"].between(0, 1).all()
