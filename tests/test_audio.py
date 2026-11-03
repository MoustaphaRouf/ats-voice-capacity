import numpy as np
import pytest

pytest.importorskip("librosa")
from atsvc.audio import FEATURE_COLUMNS, extract_features, segment_vad


def tone(sec, sr, f=200.0):
    t = np.arange(int(sec * sr)) / sr
    return 0.3 * np.sin(2 * np.pi * f * t)


def test_vad_handles_recording_that_starts_with_speech():
    # Regression: the original edge pairing shifted segments when a file began mid-transmission.
    sr = 16000
    rng = np.random.default_rng(0)
    y = np.concatenate([tone(1.0, sr), np.zeros(sr), tone(1.0, sr), np.zeros(sr)])
    y = (y + 0.001 * rng.standard_normal(len(y))).astype(np.float32)
    segs = segment_vad(y, sr)
    assert len(segs) == 2
    assert segs[0].start_s == pytest.approx(0.0, abs=0.03)
    assert segs[0].end_s == pytest.approx(1.0, abs=0.03)
    assert segs[1].start_s == pytest.approx(2.0, abs=0.03)


def test_features_fixed_length_and_short_segments_rejected():
    sr = 16000
    feats = extract_features(tone(1.0, sr).astype(np.float32), sr)
    assert list(feats) == FEATURE_COLUMNS
    assert extract_features(tone(0.1, sr).astype(np.float32), sr) is None
