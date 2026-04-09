import numpy as np
import pytest

SR = 16000

@pytest.fixture
def sr():
    return SR

@pytest.fixture
def silence():
    return np.zeros(SR, dtype=np.float32)

@pytest.fixture
def loud_sine():
    t = np.linspace(0, 1, SR, endpoint=False)
    return 0.9 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

@pytest.fixture
def quiet_sine():
    t = np.linspace(0, 1, SR, endpoint=False)
    return (0.003 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)