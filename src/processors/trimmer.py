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
        
        is_stereo = audio.ndim == 2
        original_length = audio.shape[-1] if is_stereo else len(audio)

        # Use mono mix to detect trim boundaries
        mono_ref = np.mean(audio, axis=0) if is_stereo else audio
        _, index = librosa.effects.trim(mono_ref, top_db=self.top_db)

        # Apply same trim to all channels
        trimmed = audio[:, index[0]:index[1]] if is_stereo else audio[index[0]:index[1]]
        trimmed_length = trimmed.shape[-1] if is_stereo else len(trimmed)
        
        stats = {
            'original_samples': original_length,
            'trimmed_samples': trimmed_length,
            'samples_removed': original_length - trimmed_length,
            'original_duration': original_length / sr,
            'trimmed_duration': trimmed_length / sr,
            'trim_start': index[0],
            'trim_end': index[1]
        }
        
        print(f"   Removed {stats['samples_removed']} samples "
              f"({stats['original_duration'] - stats['trimmed_duration']:.2f}s)")
        
        return trimmed, stats
