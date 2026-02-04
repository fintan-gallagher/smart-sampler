"""Audio I/O utilities"""
import numpy as np
import librosa
import soundfile as sf
from ..config import SAMPLE_RATE


def load_audio(filepath: str, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """
    Load audio file.
    
    Args:
        filepath: Path to audio file
        sr: Target sample rate
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    audio, sample_rate = librosa.load(filepath, sr=sr)
    return audio, sample_rate


def save_audio(filepath: str, audio: np.ndarray, sr: int = SAMPLE_RATE):
    """
    Save audio to file.
    
    Args:
        filepath: Output path
        audio: Audio samples
        sr: Sample rate
    """
    sf.write(filepath, audio, sr)
