import numpy as np
import pytest
from src.processors.pitch_detector import PitchDetector

SR = 16000

@pytest.fixture
def detector():
    return PitchDetector()

def make_sine(freq, sr=SR, duration=2.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)

def test_known_pitch_detected(detector):
    audio = make_sine(440)
    pitch, stats = detector.detect(audio, SR)
    assert pitch is not None
    assert pytest.approx(pitch, rel=0.05) == 440.0  # within 5%

def test_note_name_returned(detector):
    audio = make_sine(440)
    _, stats = detector.detect(audio, SR)
    assert stats['note_name'] == 'A4'

def test_noise_has_low_voiced_ratio(detector):
    rng = np.random.default_rng(0)
    noise = rng.uniform(-0.1, 0.1, SR * 2).astype(np.float32)
    _, stats = detector.detect(noise, SR)
    assert stats['voiced_ratio'] < 0.2

def test_voiced_ratio_between_0_and_1(detector):
    audio = make_sine(220)
    _, stats = detector.detect(audio, SR)
    assert 0.0 <= stats['voiced_ratio'] <= 1.0

def test_stats_always_returned_on_no_pitch(detector):
    silence = np.zeros(SR * 2, dtype=np.float32)
    pitch, stats = detector.detect(silence, SR)
    assert pitch is None
    assert 'total_frames' in stats
    assert 'voiced_frames' in stats