"""Audio normalization module"""
import numpy as np
from ..config import NORMALIZE_PEAK


class AudioNormalizer:
    """Handles audio amplitude normalization"""
    
    def __init__(self, target_peak: float = NORMALIZE_PEAK):
        self.target_peak = target_peak
    
    def normalize(self, audio: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Normalize audio to target peak amplitude.
        
        Args:
            audio: Audio samples as numpy array
            
        Returns:
            Tuple of (normalized_audio, stats_dict)
        """
        print("📊 Normalizing volume...")
        
        original_peak = np.abs(audio).max()
        
        if original_peak > 0:
            normalized = audio / original_peak * self.target_peak
        else:
            normalized = audio
        
        stats = {
            'original_peak': original_peak,
            'normalized_peak': np.abs(normalized).max(),
            'gain_applied': self.target_peak / original_peak if original_peak > 0 else 1.0
        }
        
        print(f"   Peak: {original_peak:.4f} → {stats['normalized_peak']:.4f}")
        
        return normalized, stats
