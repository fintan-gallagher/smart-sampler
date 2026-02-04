"""Pitch detection module"""
import numpy as np
import librosa
from ..config import PITCH_FMIN, PITCH_FMAX


class PitchDetector:
    """Handles pitch detection using pYIN algorithm"""
    
    def __init__(self, fmin: str = PITCH_FMIN, fmax: str = PITCH_FMAX):
        self.fmin = librosa.note_to_hz(fmin)
        self.fmax = librosa.note_to_hz(fmax)
    
    def detect(self, audio: np.ndarray, sr: int) -> tuple[float | None, dict]:
        """
        Detect fundamental frequency of audio.
        
        Args:
            audio: Audio samples as numpy array
            sr: Sample rate
            
        Returns:
            Tuple of (median_pitch_hz, stats_dict)
        """
        print("🎵 Detecting pitch...")
        
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=sr
        )
        
        voiced_pitches = f0[voiced_flag]
        
        stats = {
            'total_frames': len(f0),
            'voiced_frames': len(voiced_pitches),
            'voiced_ratio': len(voiced_pitches) / len(f0) if len(f0) > 0 else 0
        }
        
        if len(voiced_pitches) == 0:
            print("   ⚠️ No pitch detected (possible noise or percussion)")
            stats['detected_pitch'] = None
            stats['note_name'] = None
            return None, stats
        
        median_pitch = float(np.nanmedian(voiced_pitches))
        note_name = librosa.hz_to_note(median_pitch)
        
        stats['detected_pitch'] = median_pitch
        stats['note_name'] = note_name
        
        print(f"   Detected pitch: {median_pitch:.2f} Hz ({note_name})")
        
        return median_pitch, stats
