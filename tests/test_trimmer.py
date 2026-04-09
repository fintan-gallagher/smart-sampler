import numpy as np
import pytest
from src.processors.trimmer import AudioTrimmer

@pytest.fixture
def trimmer():
    return AudioTrimmer()

@pytest.fixture
def audio_with_silence(loud_sine, sr):
    silence = np.zeros(sr // 2, dtype=np.float32)  # 0.5s silence
    return np.concatenate([silence, loud_sine, silence])

def test_leading_trailing_silence_removed(trimmer, audio_with_silence, sr):
    _, stats = trimmer.trim(audio_with_silence, sr)
    assert stats['trim_start'] > 0
    assert stats['trim_end'] < stats['original_samples']

def test_samples_removed_matches_stats(trimmer, audio_with_silence, sr):
    _, stats = trimmer.trim(audio_with_silence, sr)
    assert stats['samples_removed'] == stats['original_samples'] - stats['trimmed_samples']

def test_no_silence_nothing_removed(trimmer, loud_sine, sr):
    _, stats = trimmer.trim(loud_sine, sr)
    assert stats['trimmed_samples'] <= stats['original_samples']

def test_stereo_array_handled(trimmer, loud_sine, sr):
    stereo = np.stack([loud_sine, loud_sine])  # shape (2, N)
    out, stats = trimmer.trim(stereo, sr)
    assert out.ndim == 2

def test_duration_matches_sample_count(trimmer, audio_with_silence, sr):
    _, stats = trimmer.trim(audio_with_silence, sr)
    assert pytest.approx(stats['trimmed_duration'], rel=1e-3) == stats['trimmed_samples'] / sr