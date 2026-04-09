import numpy as np
import pytest
from src.processors.normalizer import AudioNormalizer

@pytest.fixture
def norm():
    return AudioNormalizer()

def test_loud_audio_peaks_near_target(norm, loud_sine):
    out, stats = norm.normalize(loud_sine)
    assert pytest.approx(stats['normalized_peak'], abs=0.01) == 0.99

def test_gain_applied_is_correct(norm, loud_sine):
    out, stats = norm.normalize(loud_sine)
    expected_gain = 0.99 / stats['original_peak']
    assert pytest.approx(stats['gain_applied'], rel=1e-3) == expected_gain

def test_quiet_audio_capped_at_2x(norm, quiet_sine):
    _, stats = norm.normalize(quiet_sine)
    assert stats['gain_applied'] <= 2.0 + 1e-6

def test_quiet_audio_not_fully_normalised(norm, quiet_sine):
    out, stats = norm.normalize(quiet_sine)
    assert stats['normalized_peak'] < 0.5

def test_silence_does_not_crash(norm, silence):
    out, stats = norm.normalize(silence)
    assert stats['gain_applied'] == 1.0

def test_original_dbfs_in_stats(norm, loud_sine):
    _, stats = norm.normalize(loud_sine)
    assert 'original_dbfs' in stats
    assert stats['original_dbfs'] < 0  # always negative for sub-peak audio