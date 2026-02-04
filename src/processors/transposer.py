"""Pitch transposition module"""
import numpy as np
import librosa
from ..config import TRANSPOSE_THRESHOLD


class PitchTransposer:
    """Handles pitch transposition to nearest C note"""
    
    # All C notes from C2 to C7
    C_NOTES = {
        'C2': librosa.note_to_hz('C2'),
        'C3': librosa.note_to_hz('C3'),
        'C4': librosa.note_to_hz('C4'),
        'C5': librosa.note_to_hz('C5'),
        'C6': librosa.note_to_hz('C6'),
        'C7': librosa.note_to_hz('C7'),
    }
    
    def __init__(self, threshold: float = TRANSPOSE_THRESHOLD):
        self.threshold = threshold
    
    def find_nearest_c(self, frequency: float) -> tuple[str, float]:
        """
        Find the nearest C note to given frequency.
        
        Args:
            frequency: Input frequency in Hz
            
        Returns:
            Tuple of (note_name, frequency_hz)
        """
        closest = min(self.C_NOTES.items(), key=lambda x: abs(x[1] - frequency))
        print(f"   Nearest C note: {closest[0]} ({closest[1]:.2f} Hz)")
        return closest
    
    def transpose(self, audio: np.ndarray, sr: int, 
                  detected_pitch: float | None) -> tuple[np.ndarray, float | None, dict]:
        """
        Transpose audio to nearest C note.
        
        Args:
            audio: Audio samples as numpy array
            sr: Sample rate
            detected_pitch: Detected pitch in Hz, or None
            
        Returns:
            Tuple of (transposed_audio, target_frequency, stats_dict)
        """
        stats = {
            'original_pitch': detected_pitch,
            'transposed': False,
            'semitones_shifted': 0.0
        }
        
        if detected_pitch is None:
            print("   ⚠️ Skipping pitch transposition (no pitch detected)")
            stats['target_pitch'] = None
            stats['target_note'] = None
            return audio, None, stats
        
        note_name, target_freq = self.find_nearest_c(detected_pitch)
        semitones = 12 * np.log2(target_freq / detected_pitch)
        
        stats['target_pitch'] = target_freq
        stats['target_note'] = note_name
        stats['semitones_shifted'] = semitones
        
        # Skip if already close
        if abs(semitones) < self.threshold:
            print(f"   Already close to {note_name} - skipping transpose")
            return audio, target_freq, stats
        
        print(f"🎹 Transposing by {semitones:+.2f} semitones to {note_name}")
        
        transposed = librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)
        stats['transposed'] = True
        
        return transposed, target_freq, stats
