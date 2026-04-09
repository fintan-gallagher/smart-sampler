import numpy as np
import pytest
from src.processors.high_pass_filter import HighPassFilter

SR = 16000

@pytest.fixture
def hpf():
    return HighPassFilter()

def make_sine(freq, sr=SR, duration=1.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)

def rms(audio):
    return float(np.sqrt(np.mean(audio ** 2)))

def test_low_freq_attenuated(hpf):
    low = make_sine(30)  # well below 80 Hz cutoff
    out, stats = hpf.apply(low, SR)
    assert rms(out) < rms(low) * 0.5

def test_high_freq_passes_through(hpf):
    high = make_sine(1000)  # well above 80 Hz cutoff
    out, stats = hpf.apply(high, SR)
    assert rms(out) > rms(high) * 0.9

def test_rms_reduction_db_is_negative_for_low_freq(hpf):
    low = make_sine(30)
    _, stats = hpf.apply(low, SR)
    assert stats['rms_reduction_db'] < 0

def test_cutoff_too_high_skips_filter(hpf):
    audio = make_sine(440, sr=100)  # tiny sample rate → cutoff exceeds Nyquist
    out, stats = hpf.apply(audio, sr=100)
    assert stats['filtered'] is False
    assert stats['reason'] == 'cutoff_too_high'

def test_output_dtype_matches_input(hpf):
    audio = make_sine(440)
    out, _ = hpf.apply(audio, SR)
    assert out.dtype == audio.dtype