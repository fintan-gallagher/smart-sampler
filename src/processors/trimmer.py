"""Audio trimming module"""
import numpy as np
import librosa
from ..config import TRIM_TOP_DB


class AudioTrimmer:
    """Handles silence trimming from audio"""
    
    def __init__(self, top_db: float = TRIM_TOP_DB):
        self.top_db = top_db
    
    def trim(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Trim silence from beginning and end of audio.
        
        Args:
            audio: Audio samples as numpy array
            sr: Sample rate
            
        Returns:
            Tuple of (trimmed_audio, stats_dict)
        """
        print("✂️ Trimming silence...")
        
        original_length = len(audio)
        trimmed, index = librosa.effects.trim(audio, top_db=self.top_db)
        
        stats = {
            'original_samples': original_length,
            'trimmed_samples': len(trimmed),
            'samples_removed': original_length - len(trimmed),
            'original_duration': original_length / sr,
            'trimmed_duration': len(trimmed) / sr,
            'trim_start': index[0],
            'trim_end': index[1]
        }
        
        print(f"   Removed {stats['samples_removed']} samples "
              f"({stats['original_duration'] - stats['trimmed_duration']:.2f}s)")
        
        return trimmed, stats
