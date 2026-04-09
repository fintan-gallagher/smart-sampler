import numpy as np
import pytest
import librosa
from unittest.mock import patch, MagicMock
from src.processors.classifier import AudioClassifier

SR = 16000
FIXTURES = "tests/fixtures"

KICK_LABELS = {"Bass drum", "Drum", "Drum kit", "Snare drum", "Drum machine", "Drum roll"}
HARPSICHORD_LABELS = {"Harpsichord", "Piano", "Keyboard (musical)", "Electric piano", "Plucked string instrument"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_fake_infer(num_classes=521, top_class_score=0.9):
    scores = np.zeros(num_classes, dtype=np.float32)
    scores[0] = top_class_score
    mock = MagicMock()
    mock_tensor = MagicMock()
    mock_tensor.numpy.return_value = scores[np.newaxis, :]
    mock.return_value = {'output_0': mock_tensor}
    return mock

def make_fake_model(num_classes=521, top_class_score=0.9):
    model = MagicMock()
    model.signatures = {'serving_default': make_fake_infer(num_classes, top_class_score)}
    return model

FAKE_CLASS_NAMES = [f"Class_{i}" for i in range(521)]

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_singleton():
    AudioClassifier._instance = None
    yield
    AudioClassifier._instance = None

@pytest.fixture(scope="session")
def real_classifier():
    clf = AudioClassifier()
    clf._load_model()
    return clf

@pytest.fixture
def mock_classifier():
    return AudioClassifier()

@pytest.fixture(scope="session")
def kick_audio():
    audio, _ = librosa.load(f"{FIXTURES}/kick.wav", sr=SR)
    return audio

@pytest.fixture(scope="session")
def ambience_audio():
    audio, _ = librosa.load(f"{FIXTURES}/ambience.wav", sr=SR)
    return audio

@pytest.fixture(scope="session")
def harpsichord_audio():
    audio, _ = librosa.load(f"{FIXTURES}/harpsichord.wav", sr=SR)
    return audio

# ── Mock tests (ambience injection logic) ─────────────────────────────────────

def test_ambience_injected_when_quiet_and_low_confidence(mock_classifier):
    with patch('tensorflow.saved_model.load', return_value=make_fake_model(top_class_score=0.3)), \
         patch.object(mock_classifier, '_load_class_names',
                      lambda: setattr(mock_classifier, '_class_names', FAKE_CLASS_NAMES)):
        audio = np.zeros(SR, dtype=np.float32)
        results = mock_classifier.classify(audio, sr=SR, original_dbfs=-50.0)
        assert results[0] == ("Ambience", 1.0)

def test_ambience_not_injected_when_loud(mock_classifier):
    with patch('tensorflow.saved_model.load', return_value=make_fake_model(top_class_score=0.3)), \
         patch.object(mock_classifier, '_load_class_names',
                      lambda: setattr(mock_classifier, '_class_names', FAKE_CLASS_NAMES)):
        audio = np.zeros(SR, dtype=np.float32)
        results = mock_classifier.classify(audio, sr=SR, original_dbfs=-10.0)
        assert results[0][0] != "Ambience"

def test_ambience_not_injected_when_high_confidence(mock_classifier):
    with patch('tensorflow.saved_model.load', return_value=make_fake_model(top_class_score=0.9)), \
         patch.object(mock_classifier, '_load_class_names',
                      lambda: setattr(mock_classifier, '_class_names', FAKE_CLASS_NAMES)):
        audio = np.zeros(SR, dtype=np.float32)
        results = mock_classifier.classify(audio, sr=SR, original_dbfs=-50.0)
        assert results[0][0] != "Ambience"

# ── Real model tests ──────────────────────────────────────────────────────────

def test_kick_classified_as_drum(real_classifier, kick_audio):
    results = real_classifier.classify(kick_audio, SR)
    top_labels = {label for label, _ in results}
    print(f"\n Kick predictions: {results}")
    assert top_labels & KICK_LABELS, f"Expected a drum-related label in top results, got: {results}"

def test_kick_confidence_above_zero(real_classifier, kick_audio):
    results = real_classifier.classify(kick_audio, SR)
    assert results[0][1] > 0.0

def test_harpsichord_classified_correctly(real_classifier, harpsichord_audio):
    results = real_classifier.classify(harpsichord_audio, SR)
    top_labels = {label for label, _ in results}
    print(f"\n Harpsichord predictions: {results}")
    assert top_labels & HARPSICHORD_LABELS, f"Expected a keyboard/harpsichord label in top results, got: {results}"

def test_harpsichord_top_confidence_reasonable(real_classifier, harpsichord_audio):
    results = real_classifier.classify(harpsichord_audio, SR)
    assert results[0][1] > 0.1

def test_ambience_triggers_injection(real_classifier, ambience_audio):
    rms = float(np.sqrt(np.mean(ambience_audio ** 2)))
    dbfs = 20 * np.log10(rms) if rms > 0 else -120.0
    results = real_classifier.classify(ambience_audio, SR, original_dbfs=dbfs)
    print(f"\n   Ambience dBFS: {dbfs:.1f}")
    print(f"   Ambience predictions: {results}")
    if dbfs < -30.0:
        assert results[0][0] == "Ambience"

def test_top_k_respected(real_classifier, kick_audio):
    results = real_classifier.classify(kick_audio, SR, top_k=3)
    assert len(results) == 3

def test_results_are_label_confidence_tuples(real_classifier, kick_audio):
    results = real_classifier.classify(kick_audio, SR)
    for label, conf in results:
        assert isinstance(label, str)
        assert isinstance(conf, float)