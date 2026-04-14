"""Audio trimming module"""
import numpy as np
import librosa
from ..config import TRIM_TOP_DB


class AudioTrimmer:
    """Handles silence trimming from audio"""
    
    def __init__(self, top_db: float = TRIM_TOP_DB,
                 pad_ms: float = 30.0, tail_ms: float = 80.0,
                 frame_length: int = 1024, hop_length: int = 256):
        self.top_db = top_db
        self.pad_ms = pad_ms
        self.tail_ms = tail_ms
        self.frame_length = frame_length
        self.hop_length = hop_length

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

        # Adaptive threshold: use the quietest 10% of frames as the noise floor,
        # then trim anything within top_db of that floor.
        rms = librosa.feature.rms(y=mono_ref,
                                   frame_length=self.frame_length,
                                   hop_length=self.hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms)
        noise_floor = np.percentile(rms_db, 10)
        threshold = noise_floor + self.top_db

        # Find first and last frame above the threshold
        above = np.where(rms_db > threshold)[0]

        if len(above) == 0:
            # Everything is below threshold — return as-is
            trimmed = audio
            start_sample = 0
            end_sample = original_length
        else:
            first_frame = above[0]
            last_frame = above[-1]

            # Convert frame indices to sample indices
            start_sample = first_frame * self.hop_length
            end_sample = min((last_frame + 1) * self.hop_length + self.frame_length,
                             original_length)

            # Add padding so we don't clip the attack or tail
            pad_samples = int(sr * self.pad_ms / 1000)
            tail_samples = int(sr * self.tail_ms / 1000)
            start_sample = max(0, start_sample - pad_samples)
            end_sample = min(original_length, end_sample + tail_samples)

        # Apply same trim to all channels
        trimmed = (audio[:, start_sample:end_sample] if is_stereo
                   else audio[start_sample:end_sample])
        trimmed_length = trimmed.shape[-1] if is_stereo else len(trimmed)
        
        stats = {
            'original_samples': original_length,
            'trimmed_samples': trimmed_length,
            'samples_removed': original_length - trimmed_length,
            'original_duration': original_length / sr,
            'trimmed_duration': trimmed_length / sr,
            'trim_start': start_sample,
            'trim_end': end_sample
        }
        
        print(f"   Removed {stats['samples_removed']} samples "
              f"({stats['original_duration'] - stats['trimmed_duration']:.2f}s)")
        
        return trimmed, stats
