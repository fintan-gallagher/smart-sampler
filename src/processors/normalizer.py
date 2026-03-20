"""Audio normalization module"""
import numpy as np
from ..config import NORMALIZE_PEAK

MAX_GAIN_DB = 20.0  # Limit gain to prevent excessive amplification
QUIET_DBFS    = -30.0  # below this, treat as quiet/ambient
QUIET_GAIN_DB = 6.0    # quiet audio gets at most this much boost (2x)


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
        original_rms = float(np.sqrt(np.mean(audio**2)))
        original_dbfs = 20 * np.log10(original_rms) if original_rms > 0 else -120.0

        MAX_GAIN_DB   = 20.0
        QUIET_DBFS    = -30.0
        QUIET_GAIN_DB = 6.0
        
        if original_peak > 0:
            ideal_gain  = self.target_peak / original_peak
        
            if original_dbfs < QUIET_DBFS:
                max_gain    = 10 ** (QUIET_GAIN_DB / 20)
            else:
                max_gain    = 10 ** (MAX_GAIN_DB / 20)

            gain        = min(ideal_gain, max_gain)
            normalized  = audio * gain
        else:
            gain = 1.0
            normalized = audio
        
        stats = {
            'original_peak': original_peak,
            'normalized_peak': np.abs(normalized).max(),
            'gain_applied': gain,
            'original_dbfs': original_dbfs,
        }
        
        print(f"   Peak: {original_peak:.4f} → {stats['normalized_peak']:.4f}"
              f"(gain {gain:.2f}x, pre-norm {original_dbfs:.1f} dBFS)")
        
        return normalized, stats
