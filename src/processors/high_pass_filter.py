import numpy as np
from scipy import signal
from ..config import HPF_CUTOFF, HPF_ORDER


class HighPassFilter:
    """Applies high pass filtering to remove low-frequency noise"""
    
    def __init__(self, cutoff: float = HPF_CUTOFF, order: int = HPF_ORDER):
        """
        Args:
            cutoff: Cutoff frequency in Hz
            order: Filter order (higher = steeper rolloff)
        """
        self.cutoff = cutoff
        self.order = order
    
    def apply(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Apply high pass filter to audio.
        
        Args:
            audio: Audio samples as numpy array
            sr: Sample rate
            
        Returns:
            Tuple of (filtered_audio, stats_dict)
        """
        print(f"🔊 Applying high pass filter ({self.cutoff}Hz cutoff)...")
        
        # Design Butterworth high pass filter
        nyquist = sr / 2
        normalized_cutoff = self.cutoff / nyquist
        
        # Ensure cutoff is valid
        if normalized_cutoff >= 1:
            print(f"   Warning: Cutoff {self.cutoff}Hz too high for sample rate {sr}Hz, skipping filter")
            return audio, {'filtered': False, 'reason': 'cutoff_too_high'}
        
        b, a = signal.butter(self.order, normalized_cutoff, btype='high')
        
        # Apply filter (filtfilt for zero phase distortion)
        filtered = signal.filtfilt(b, a, audio)
        
        # Measure noise reduction
        original_rms = np.sqrt(np.mean(audio**2))
        filtered_rms = np.sqrt(np.mean(filtered**2))
        
        stats = {
            'filtered': True,
            'cutoff_hz': self.cutoff,
            'order': self.order,
            'original_rms': original_rms,
            'filtered_rms': filtered_rms,
            'rms_reduction_db': 20 * np.log10(filtered_rms / original_rms) if original_rms > 0 else 0
        }
        
        print(f"   RMS change: {stats['rms_reduction_db']:.2f}dB")
        
        return filtered.astype(audio.dtype), stats