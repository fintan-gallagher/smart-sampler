import numpy as np
import pytest
from src.processors.trimmer import AudioTrimmer

@pytest.fixture
def trimmer():
    return AudioTrimmer()

@pytest.fixture
def audio_with_silence(loud_sine, sr):
    noise = np.random.default_rng(0).normal(0, 0.001, sr // 2).astype(np.float32)
    return np.concatenate([noise, loud_sine, noise])

def test_leading_trailing_silence_removed(trimmer, audio_with_silence, sr):
    _, stats = trimmer.trim(audio_with_silence, sr)
    assert stats['samples_removed'] > 0

def test_samples_removed_matches_stats(trimmer, audio_with_silence, sr):
    _, stats = trimmer.trim(audio_with_silence, sr)
    assert stats['samples_removed'] == stats['original_samples'] - stats['trimmed_samples']

def test_trimmed_shorter_than_original(trimmer, audio_with_silence, sr):
    _, stats = trimmer.trim(audio_with_silence, sr)
    assert stats['trimmed_samples'] < stats['original_samples']

def test_loud_audio_mostly_kept(trimmer, loud_sine, sr):
    out, stats = trimmer.trim(loud_sine, sr)
    # Loud sine has no silence — most of it should survive
    assert stats['trimmed_samples'] > stats['original_samples'] * 0.8

def test_stereo_array_handled(trimmer, loud_sine, sr):
    stereo = np.stack([loud_sine, loud_sine])
    out, stats = trimmer.trim(stereo, sr)
    assert out.ndim == 2

def test_duration_matches_sample_count(trimmer, audio_with_silence, sr):
    _, stats = trimmer.trim(audio_with_silence, sr)
    assert pytest.approx(stats['trimmed_duration'], rel=1e-3) == stats['trimmed_samples'] / sr

def test_silence_returns_full_audio(trimmer, silence, sr):
    # Pure silence — nothing above threshold, should return as-is
    out, stats = trimmer.trim(silence, sr)
    assert stats['trimmed_samples'] == stats['original_samples']