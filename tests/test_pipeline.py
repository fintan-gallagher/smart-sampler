import os
import pytest
import numpy as np
import soundfile as sf
from src.main import SmartSampler

FIXTURES = "tests/fixtures"
SR = 16000

@pytest.fixture(scope="session")
def sampler():
    return SmartSampler()

@pytest.fixture
def output_path(tmp_path):
    return str(tmp_path / "output.wav")

def test_pipeline_produces_output(sampler, output_path):
    results = sampler.process(f"{FIXTURES}/kick.wav", output_path)
    assert os.path.exists(output_path)
    audio, sr = sf.read(output_path)
    assert len(audio) > 0
    assert sr > 0

def test_pipeline_returns_expected_keys(sampler, output_path):
    results = sampler.process(f"{FIXTURES}/kick.wav", output_path)
    assert 'predictions' in results
    assert 'detected_pitch' in results
    assert 'raw_audio' in results
    assert 'clean_audio' in results
    assert 'sample_rate' in results

def test_processed_audio_trimmed(sampler, output_path):
    results = sampler.process(f"{FIXTURES}/kick.wav", output_path)
    assert len(results['clean_audio']) <= len(results['raw_audio'])

def test_processed_audio_normalised(sampler, output_path):
    results = sampler.process(f"{FIXTURES}/kick.wav", output_path)
    peak = np.abs(results['clean_audio']).max()
    assert peak > 0.5  # should be close to 0.99 for a loud kick

def test_pipeline_produces_predictions(sampler, output_path):
    results = sampler.process(f"{FIXTURES}/kick.wav", output_path)
    assert len(results['predictions']) > 0
    label, conf = results['predictions'][0]
    assert isinstance(label, str)
    assert conf > 0.0

def test_pipeline_detects_pitch_for_tonal(sampler, output_path):
    results = sampler.process(f"{FIXTURES}/harpsichord.wav", output_path)
    assert results['detected_pitch'] is not None
    assert results['detected_pitch'] > 0

def test_pipeline_pitch_may_be_none_for_percussion(sampler, output_path):
    results = sampler.process(f"{FIXTURES}/kick.wav", output_path)
    # Kick may or may not have a detected pitch — just check it doesn't crash
    assert 'detected_pitch' in results

def test_ambience_classified_through_pipeline(sampler, output_path):
    results = sampler.process(f"{FIXTURES}/ambience.wav", output_path)
    labels = [label for label, _ in results['predictions']]
    print(f"\n   Ambience pipeline predictions: {results['predictions']}")
    # If the recording is genuinely quiet, Ambience should appear
    rms = float(np.sqrt(np.mean(results['raw_audio'] ** 2)))
    dbfs = 20 * np.log10(rms) if rms > 0 else -120.0
    if dbfs < -30.0:
        assert labels[0] == "Ambience"

def test_output_is_valid_audio(sampler, output_path):
    results = sampler.process(f"{FIXTURES}/kick.wav", output_path)
    audio, sr = sf.read(output_path)
    assert audio.ndim >= 1
    assert not np.any(np.isnan(audio))
    assert not np.any(np.isinf(audio))
