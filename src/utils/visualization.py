"""Visualization utilities"""
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..config import SAMPLE_RATE


class SpectrogramPlotter:
    """Generates spectrogram comparisons"""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sr = sample_rate
    
    def plot_comparison(self, raw_audio: np.ndarray, clean_audio: np.ndarray,
                        raw_pitch: float | None, clean_pitch: float | None,
                        output_path: str):
        """
        Generate side-by-side spectrogram comparison.
        
        Args:
            raw_audio: Original audio samples
            clean_audio: Processed audio samples
            raw_pitch: Detected pitch of original (Hz or None)
            clean_pitch: Target pitch after processing (Hz or None)
            output_path: Output image path
        """
        print("📊 Generating spectrograms...")
        
        # Compute mel spectrograms
        S_raw = librosa.feature.melspectrogram(y=raw_audio, sr=self.sr, n_mels=128, fmax=8000)
        S_clean = librosa.feature.melspectrogram(y=clean_audio, sr=self.sr, n_mels=128, fmax=8000)
        
        S_raw_db = librosa.power_to_db(S_raw, ref=np.max)
        S_clean_db = librosa.power_to_db(S_clean, ref=np.max)
        
        # Create figure
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        
        # Row 1: Waveforms
        self._plot_waveform(axes[0, 0], raw_audio, raw_pitch, 'Raw Waveform')
        self._plot_waveform(axes[0, 1], clean_audio, clean_pitch, 'Processed Waveform', color='green')
        
        # Row 2: Spectrograms
        self._plot_spectrogram(axes[1, 0], S_raw_db, raw_pitch, 'Raw Spectrogram', fig)
        self._plot_spectrogram(axes[1, 1], S_clean_db, clean_pitch, 'Processed Spectrogram', fig)
        
        # Row 3: Frequency analysis and stats
        self._plot_frequency_comparison(axes[2, 0], S_raw_db, S_clean_db, raw_pitch, clean_pitch)
        self._plot_stats(axes[2, 1], raw_audio, clean_audio, raw_pitch, clean_pitch)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Spectrogram saved as {output_path}")
    
    def _plot_waveform(self, ax, audio, pitch, title, color='blue'):
        """Plot waveform with pitch annotation"""
        times = np.linspace(0, len(audio) / self.sr, len(audio))
        ax.plot(times, audio, linewidth=0.5, color=color)
        
        pitch_text = f"{pitch:.2f} Hz ({librosa.hz_to_note(pitch)})" if pitch else "No pitch detected"
        ax.set_title(f'{title}\nPitch: {pitch_text}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
    
    def _plot_spectrogram(self, ax, S_db, pitch, title, fig):
        """Plot mel spectrogram with pitch line"""
        img = librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=self.sr,
                                        fmax=8000, ax=ax, cmap='viridis')
        ax.set_title(title, fontsize=12, fontweight='bold')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
        if pitch and pitch <= 8000:
            ax.axhline(y=pitch, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    def _plot_frequency_comparison(self, ax, S_raw_db, S_clean_db, raw_pitch, clean_pitch):
        """Plot frequency content comparison"""
        raw_freq_avg = np.mean(S_raw_db, axis=1)
        clean_freq_avg = np.mean(S_clean_db, axis=1)
        mel_freqs = librosa.mel_frequencies(n_mels=128, fmax=8000)
        
        ax.plot(mel_freqs, raw_freq_avg, label='Raw', linewidth=2)
        ax.plot(mel_freqs, clean_freq_avg, label='Processed', linewidth=2, alpha=0.7)
        
        if raw_pitch and raw_pitch <= 8000:
            ax.axvline(x=raw_pitch, color='red', linestyle='--', alpha=0.7)
        if clean_pitch and clean_pitch <= 8000:
            ax.axvline(x=clean_pitch, color='lime', linestyle='--', alpha=0.7)
        
        ax.set_title('Average Frequency Content', fontsize=12, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (dB)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    def _plot_stats(self, ax, raw_audio, clean_audio, raw_pitch, clean_pitch):
        """Plot processing statistics"""
        raw_pitch_text = f"{raw_pitch:.2f} Hz" if raw_pitch else "N/A"
        clean_pitch_text = f"{clean_pitch:.2f} Hz" if clean_pitch else "N/A"
        
        stats_text = f"""Processing Statistics:

Raw Audio:
• Duration: {len(raw_audio)/self.sr:.2f}s
• Peak: {np.abs(raw_audio).max():.4f}
• RMS: {np.sqrt(np.mean(raw_audio**2)):.4f}
• Pitch: {raw_pitch_text}

Processed Audio:
• Duration: {len(clean_audio)/self.sr:.2f}s
• Peak: {np.abs(clean_audio).max():.4f}
• RMS: {np.sqrt(np.mean(clean_audio**2)):.4f}
• Pitch: {clean_pitch_text}

Changes:
• Trimmed: {len(raw_audio) - len(clean_audio)} samples
• Time saved: {(len(raw_audio) - len(clean_audio))/self.sr:.2f}s
"""
        
        ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.3))
        ax.axis('off')
