import os
import pytest
from src.utils.sfz_generator import SFZGenerator

FIXTURES = "tests/fixtures"

@pytest.fixture
def gen():
    return SFZGenerator()

def test_generate_with_detected_pitch(gen):
    sfz = gen.generate("kick.wav", 440.0, "Kick")
    assert "pitch_keycenter=69" in sfz
    assert "440.00 Hz" in sfz

def test_generate_without_pitch_defaults_to_c4(gen):
    sfz = gen.generate("kick.wav", None, "Kick")
    assert "pitch_keycenter=60" in sfz
    assert "defaulting to C4" in sfz

def test_generate_contains_sample_filename(gen):
    sfz = gen.generate("kick.wav", 440.0, "Kick")
    assert "sample=kick.wav" in sfz

def test_generate_contains_label_in_comment(gen):
    sfz = gen.generate("kick.wav", 440.0, "Kick drum")
    assert "// Kick drum Sample" in sfz

def test_fixed_velocity_adds_amp_veltrack(gen):
    sfz = gen.generate("kick.wav", 440.0, "Kick", fixed_velocity=True)
    assert "amp_veltrack=0" in sfz

def test_dynamic_velocity_no_amp_veltrack(gen):
    sfz = gen.generate("kick.wav", 440.0, "Kick", fixed_velocity=False)
    assert "amp_veltrack" not in sfz

def test_loop_enabled(gen):
    sfz = gen.generate("kick.wav", 440.0, "Kick", loop=True)
    assert "loop_mode=loop_sustain" in sfz
    assert "loop_start=0" in sfz

def test_loop_disabled(gen):
    sfz = gen.generate("kick.wav", 440.0, "Kick", loop=False)
    assert "loop_mode=one_shot" in sfz
    assert "loop_sustain" not in sfz

def test_loop_end_from_real_audio(gen):
    sfz = gen.generate("kick.wav", 440.0, "Kick",
                       audio_path=f"{FIXTURES}/kick.wav", loop=True)
    assert "loop_end=" in sfz
    # Extract the loop_end value and check it's a positive number
    for line in sfz.splitlines():
        if "loop_end=" in line:
            value = int(line.strip().split("=")[1])
            assert value > 0
            break

def test_save_writes_file(gen, tmp_path):
    # Create a dummy WAV so _get_sample_frames can read it
    import soundfile as sf
    import numpy as np
    dummy_wav = tmp_path / "test.wav"
    sf.write(str(dummy_wav), np.zeros(1000, dtype=np.float32), 16000)

    sfz_path = str(tmp_path / "test.sfz")
    gen.save(sfz_path, "test.wav", 440.0, "Kick")

    assert os.path.exists(sfz_path)
    with open(sfz_path) as f:
        content = f.read()
    assert "sample=test.wav" in content
    assert "pitch_keycenter=69" in content