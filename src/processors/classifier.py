"""Audio classification module using YAMNet"""
import os
import csv
import numpy as np
import librosa
import tensorflow as tf
from ..config import MODEL_PATH, SAMPLE_RATE


class AudioClassifier:
    """Handles audio classification using YAMNet"""
    
    _instance = None  # Singleton for model caching
    
    def __new__(cls):
        """Singleton pattern to cache model across calls"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._infer = None
            cls._instance._class_names = None
        return cls._instance
    
    def _load_model(self):
        """Load YAMNet model (only once)"""
        if self._model is None:
            print("🔍 Loading YAMNet model (one-time operation)...")
            self._model = tf.saved_model.load(MODEL_PATH)
            # Use the default serving signature or find available signatures
            if 'serving_default' in self._model.signatures:
                self._infer = self._model.signatures['serving_default']
            else:
                # Use the first available signature
                available_sigs = list(self._model.signatures.keys())
                if available_sigs:
                    self._infer = self._model.signatures[available_sigs[0]]
                else:
                    # Fall back to calling the model directly
                    self._infer = self._model
            self._load_class_names()
            print("✅ Model loaded successfully")
        else:
            print("🔍 Using cached YAMNet model...")
    
    def _load_class_names(self):
        """Load class names from CSV"""
        class_map_path = os.path.join(MODEL_PATH, 'assets', 'yamnet_class_map.csv')
        with open(class_map_path, newline='') as f:
            reader = csv.DictReader(f)
            self._class_names = [row["display_name"] for row in reader]
    
    def classify(self, audio: np.ndarray, sr: int, 
                 top_k: int = 3) -> list[tuple[str, float]]:
        """
        Classify audio using YAMNet.
        
        Args:
            audio: Audio samples as numpy array
            sr: Sample rate
            top_k: Number of top predictions to return
            
        Returns:
            List of (label, confidence) tuples
        """
        self._load_model()
        
        # Prepare audio
        waveform = audio
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)
        if sr != SAMPLE_RATE:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SAMPLE_RATE)
        
        # YAMNet expects 1D waveform, not batched
        if waveform.ndim > 1:
            waveform = waveform.squeeze()
        
        # Run inference
        print("🤖 Running classification...")
        try:
            outputs = self._infer(waveform=tf.constant(waveform, dtype=tf.float32))
            # Try different possible output key names
            if 'output_0' in outputs:
                scores = outputs['output_0'].numpy()
            elif 'scores' in outputs:
                scores = outputs['scores'].numpy()
            else:
                # Use the first output
                scores = list(outputs.values())[0].numpy()
        except Exception:
            # If signature doesn't work, try direct call
            outputs = self._model(tf.constant(waveform, dtype=tf.float32))
            scores = outputs[0].numpy() if isinstance(outputs, (list, tuple)) else outputs.numpy()
        
        # Get top predictions
        mean_scores = np.mean(scores, axis=0) if scores.ndim > 1 else scores
        top_indices = np.argsort(mean_scores)[::-1][:top_k]
        
        predictions = [
            (self._class_names[i] if i < len(self._class_names) else "unknown", 
             float(mean_scores[i]))
            for i in top_indices
        ]
        
        return predictions