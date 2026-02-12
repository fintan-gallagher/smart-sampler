import numpy as np
import tensorflow as tf
from ..config import DTLN_MODEL_PATH, DTLN_BLOCK_LEN, DTLN_BLOCK_SHIFT

class DTLNDenoiser:

    def __init__(self, model_path: str = DTLN_MODEL_PATH):
        """
        Args:
            model_path: Path to DTLN model directory
        """
        self.model_path = model_path
        self.block_len = DTLN_BLOCK_LEN
        self.block_shift = DTLN_BLOCK_SHIFT
        self.model = None

    def load_model(self):
        """Lazy load the DTLN model"""
        if self.model is None:
            print(f"Loading DTLN model from {self.model_path}...")
            try:
                self.model = tf.saved_model.load(self.model_path)
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Error Loading Model: {e}")
                raise

            def apply(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
                """
                Apply DTLN noise reduction.
                
                Args:
                    audio: Audio samples as numpy array
                    sr: Sample rate (must be 16kHz for DTLN)
                    
                Returns:
                    Tuple of (denoised_audio, stats_dict)
                """
                print("🔇 Applying DTLN noise reduction...")
                
                if sr != 16000:
                    print(f"   ⚠️ Warning: DTLN expects 16kHz, got {sr}Hz")
                
                # Load model if not already loaded
                self.load_model()
                
                # Measure original noise level
                original_rms = np.sqrt(np.mean(audio**2))
                
                # Process audio (DTLN expects specific input format)
                try:
                    # Add batch dimension and convert to float32
                    audio_input = audio.astype(np.float32).reshape(1, -1)
                    
                    # Run inference
                    denoised = self.model(audio_input)
                    denoised = denoised.numpy().flatten()
                    
                    # Measure denoised level
                    denoised_rms = np.sqrt(np.mean(denoised**2))
                    
                    stats = {
                        'denoised': True,
                        'original_rms': original_rms,
                        'denoised_rms': denoised_rms,
                        'noise_reduction_db': 20 * np.log10(denoised_rms / original_rms) if original_rms > 0 else 0
                    }
                    
                    print(f"   Noise reduction: {stats['noise_reduction_db']:.2f}dB")
                    
                    return denoised.astype(audio.dtype), stats
                    
                except Exception as e:
                    print(f"   ❌ DTLN processing failed: {e}")
                    return audio, {'denoised': False, 'error': str(e)}