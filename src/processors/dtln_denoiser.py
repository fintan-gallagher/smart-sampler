import numpy as np
import tensorflow as tf
import os
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
        self.interp_1 = None
        self.interp_2 = None

    def load_model(self):
        """Lazy load the DTLN model"""
        if self.interp_1 is None:
            print(f"Loading DTLN model from {self.model_path}...")
            try:
                model_1_path = os.path.join(self.model_path, 'model_1.tflite')
                model_2_path = os.path.join(self.model_path, 'model_2.tflite')

                self.interp_1 = tf.lite.Interpreter(model_path=model_1_path)
                self.interp_2 = tf.lite.Interpreter(model_path=model_2_path)

                self.interp_1.allocate_tensors()
                self.interp_2.allocate_tensors()

                self.in_det_1  = self.interp_1.get_input_details()
                self.out_det_1 = self.interp_1.get_output_details()
                self.in_det_2  = self.interp_2.get_input_details()
                self.out_det_2 = self.interp_2.get_output_details()

                print(f"Model loaded successfully.")
            except Exception as e:
                print(f"Error Loading Model: {e}")
                raise

    def _process_chunk(self, audio: np.ndarray) -> np.ndarray:
        """
        Process a 1D mono float32 audio array through DTLN block by block.
        Maintains LSTM state across blocks for continuity.
        """
        audio_f32 = audio.astype(np.float32)
        n_samples = len(audio_f32)

        # Normalise input to [-1, 1] — model was trained on normalised audio
        input_peak = np.abs(audio_f32).max()
        if input_peak > 0:
            audio_f32 = audio_f32 / input_peak

        states_1 = np.zeros(self.in_det_1[1]['shape'], dtype=np.float32)
        states_2 = np.zeros(self.in_det_2[1]['shape'], dtype=np.float32)

        out_file   = np.zeros(n_samples, dtype=np.float32)
        in_buffer  = np.zeros(self.block_len, dtype=np.float32)
        out_buffer = np.zeros(self.block_len, dtype=np.float32)

        # Exact num_blocks formula from official script
        num_blocks = (n_samples - (self.block_len - self.block_shift)) // self.block_shift

        for idx in range(num_blocks):
            # Shift and fill input buffer
            in_buffer[:-self.block_shift] = in_buffer[self.block_shift:]
            in_buffer[-self.block_shift:] = audio_f32[idx * self.block_shift:(idx * self.block_shift) + self.block_shift]

            # FFT → magnitude and phase
            in_block_fft = np.fft.rfft(in_buffer)
            in_mag   = np.reshape(np.abs(in_block_fft), (1, 1, -1)).astype(np.float32)
            in_phase = np.angle(in_block_fft)

            # === Model 1: spectral mask ===
            # Note: index [1] = states, index [0] = magnitude (matches official script)
            self.interp_1.set_tensor(self.in_det_1[1]['index'], states_1)
            self.interp_1.set_tensor(self.in_det_1[0]['index'], in_mag)
            self.interp_1.invoke()
            out_mask = self.interp_1.get_tensor(self.out_det_1[0]['index'])
            states_1 = self.interp_1.get_tensor(self.out_det_1[1]['index'])

            # Apply mask → iFFT → time domain frame
            estimated_complex = in_mag * out_mask * np.exp(1j * in_phase)
            estimated_block = np.reshape(
                np.fft.irfft(estimated_complex), (1, 1, -1)
            ).astype(np.float32)

            # === Model 2: time-domain refinement ===
            # Note: index [1] = states, index [0] = frame (matches official script)
            self.interp_2.set_tensor(self.in_det_2[1]['index'], states_2)
            self.interp_2.set_tensor(self.in_det_2[0]['index'], estimated_block)
            self.interp_2.invoke()
            out_block = self.interp_2.get_tensor(self.out_det_2[0]['index'])
            states_2  = self.interp_2.get_tensor(self.out_det_2[1]['index'])

            # Overlap-add — matches official script exactly
            out_buffer[:-self.block_shift] = out_buffer[self.block_shift:]
            out_buffer[-self.block_shift:] = np.zeros(self.block_shift)
            out_buffer += np.squeeze(out_block)
            out_file[idx * self.block_shift:(idx * self.block_shift) + self.block_shift] = out_buffer[:self.block_shift]

        # Rescale to original level
        out_file *= input_peak
        return out_file

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
            out_buffer = self._process_chunk(audio)

            denoised_rms = float(np.sqrt(np.mean(out_buffer**2)))
                    
            stats = {
                'denoised': True,
                'original_rms': float(original_rms),
                'denoised_rms': float(denoised_rms),
                'noise_reduction_db': float(20 * np.log10(denoised_rms / original_rms)) if original_rms > 0 else 0
            }
                    
            print(f"   Noise reduction: {stats['noise_reduction_db']:.2f}dB")
                    
            return out_buffer.astype(audio.dtype), stats
                    
        except Exception as e:
            print(f"   ❌ DTLN processing failed: {e}")
            return audio, {'denoised': False, 'error': str(e)}