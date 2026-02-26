"""Main entry point for Smart Sampler"""
import os
import shutil
import numpy as np
import librosa

# Optimize TensorFlow for Pi
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from .config import (
    TEST_MODE, TEST_AUDIO_PATH, RAW_FILENAME, CLEAN_FILENAME, SAMPLE_RATE
)
from .processors import (
    AudioRecorder, AudioTrimmer, AudioNormalizer,
    PitchDetector, AudioClassifier, HighPassFilter, DTLNDenoiser
)
from .utils import load_audio, save_audio, SpectrogramPlotter, FileManager, SFZGenerator


class SmartSampler:
    """Main audio processing pipeline"""
    
    def __init__(self):
        self.recorder = AudioRecorder()
        self.high_pass_filter = HighPassFilter()
        self.dtln_denoiser = None # Lazy Load
        self.trimmer = AudioTrimmer()
        self.normalizer = AudioNormalizer()
        self.pitch_detector = PitchDetector()
        self.classifier = AudioClassifier()
        self.plotter = SpectrogramPlotter()
        self.file_manager = FileManager()
        self.sfz_generator = SFZGenerator()

    def _to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert stereo [channels, samples] to mono [samples]"""
        if audio.ndim == 2:
            return np.mean(audio, axis=0)
        return audio
    
    def process(self, input_path: str, output_path: str, use_dtln = False) -> dict:
        """
        Run full processing pipeline.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save processed audio
            
        Returns:
            Dict with processing results
        """
        print("🎧 Processing audio...")
        
        # Load audio
        audio, sr = load_audio(input_path, sr=None, mono=False)
        raw_audio = audio.copy()
        channels = audio.shape[0] if audio.ndim == 2 else 1
        print(f"   Loaded: {sr}Hz | {'Stereo' if channels == 2 else 'Mono'} | {audio.shape[-1]} samples")

        # 1. High pass filter (remove low-frequency noise first)
        audio, hpf_stats = self.high_pass_filter.apply(audio, sr)

        # 2. DTLN noise reduction (optional)
        if use_dtln:
            if self.dtln_denoiser is None:
                self.dtln_denoiser = DTLNDenoiser()
            if audio.ndim == 2:
                # Process each channel independently
                denoised_channels = []
                for ch in range(audio.shape[0]):
                    ch_16k = librosa.resample(audio[ch], orig_sr=sr, target_sr=16000)
                    ch_denoised, _ = self.dtln_denoiser.apply(ch_16k, 16000)
                    ch_native = librosa.resample(ch_denoised, orig_sr=16000, target_sr=sr)
                    denoised_channels.append(ch_native)
                audio = np.stack(denoised_channels, axis=0)
            else:
                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                audio_16k, _ = self.dtln_denoiser.apply(audio_16k, 16000)
                audio = librosa.resample(audio_16k, orig_sr=16000, target_sr=sr)
            print(f"   DTLN applied, restored to {sr}Hz")
        
        # 3. Trim silence
        audio, trim_stats = self.trimmer.trim(audio, sr)
        
        # 4. Normalize (for classification)
        audio, norm_stats = self.normalizer.normalize(audio)
        
        # 5. Classify (before pitch shift for accuracy)
        mono_mix = self._to_mono(audio)
        audio_for_classification = librosa.resample(mono_mix, orig_sr=sr, target_sr=16000) if sr != 16000 else mono_mix
        predictions = self.classifier.classify(audio_for_classification, 16000)
        
        # 6. Detect pitch
        detected_pitch, pitch_stats = self.pitch_detector.detect(mono_mix, sr)
        
        # Save processed audio
        save_audio(output_path, audio, sr)
        print(f"Saved: {output_path} ({sr}Hz, {'stereo' if audio.ndim == 2 else 'mono'})")
        
        # Print stats
        self._print_stats(self._to_mono(raw_audio), mono_mix, sr, detected_pitch)
        
        return {
            'predictions': predictions,
            'detected_pitch': detected_pitch,
            'raw_audio': self._to_mono(raw_audio),
            'clean_audio': mono_mix,
            'sample_rate': sr
        }
    
    def _print_stats(self, raw, clean, sr, detected_pitch):
        """Print processing statistics"""
        print(f"\n📊 Processing Stats:")
        print(f"   Original length: {len(raw)/sr:.2f}s ({len(raw)} samples)")
        print(f"   Final length:    {len(clean)/sr:.2f}s ({len(clean)} samples)")
        print(f"   Original peak:   {abs(raw).max():.4f}")
        print(f"   Final peak:      {abs(clean).max():.4f}")
        
        if detected_pitch:
            import librosa
            note = librosa.hz_to_note(detected_pitch)
            print(f"   Detected pitch:  {note} ({detected_pitch:.2f} Hz)\n")
        else:
            print(f"   Detected pitch:  N/A (no pitch detected)\n")
    
    def run(self):
        """Run the full sampler workflow"""
        print("=" * 60)
        print("🎵 Smart Sampler - Audio Processing Pipeline")
        print("=" * 60)
        
        if TEST_MODE:
            print("\n🧪 TEST MODE - Using pre-recorded audio")
            
            #get all WAV files in directory
            test_dir = os.path.dirname(TEST_AUDIO_PATH)
            wav_files = [f for f in os.listdir(test_dir) if f.lower().endswith('wav')]

            if not wav_files:
                print(f"Error: No WAV files found in {test_dir}")
                return
            #Display available files
            print(f"Available test samples")
            for i, filename in enumerate(wav_files, 1):
                print(f"   {i}. {filename}")

            # Prompt user to select file
            while True:
                try:
                    choice = input(f"Select a file to process (1-{len(wav_files)}): ").strip()
                    idx = int(choice) - 1
                    if 0 <= idx < len(wav_files):
                        selected_file = os.path.join(test_dir, wav_files[idx])
                        print(f"Selected file: {wav_files[idx]}\n")
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(wav_files)}.")
                except (ValueError, KeyboardInterrupt):
                    print("Invalid input. Please enter a valid number.")
                    continue

            shutil.copy2(selected_file, RAW_FILENAME)
        else:
            print("\n🎙️ LIVE MODE - Recording from microphone\n")
            self.recorder.record(RAW_FILENAME)

        # DTLN User Choice
        print("\n Apply DTLN Noise Reduction? (may result in artefacting/removal of desired noise)")
        use_dtln_input = input("Use DTLN? (Y/N): ").strip().lower()
        use_dtln = use_dtln_input in ('y', 'yes')

        if use_dtln:
            print("DTLN will be applied")
        else:
            print("Skipping DTLN")
        
        # Process audio
        results = self.process(RAW_FILENAME, CLEAN_FILENAME, use_dtln)
        
        # Get label from user
        label = self.file_manager.prompt_label_selection(results['predictions'])
        
        # Save with label and generate spectrogram
        def gen_spectrogram(path):
            self.plotter.plot_comparison(
                results['raw_audio'],
                results['clean_audio'],
                results['detected_pitch'],
                results['detected_pitch'],
                path,
                sample_rate=results['sample_rate']
            )

        def gen_sfz(path, audio_filename):
            self.sfz_generator.save(
                path,
                audio_filename,
                results['detected_pitch'],
                label
            )
        
        paths = self.file_manager.save_with_label(
            CLEAN_FILENAME,
            RAW_FILENAME,
            label,
            results['detected_pitch'],
            gen_spectrogram,
            gen_sfz
        )
        
        print(f"\n💾 Saved files:")
        print(f"   RAW:         {paths['raw']}")
        print(f"   CLEAN:       {paths['clean']}")
        print(f"   SPECTROGRAM: {paths['spectrogram']}")
        print(f"   SFZ: {paths['sfz']}")
        
        print("\n" + "=" * 60)
        print("✅ Processing complete!")
        print("=" * 60)


if __name__ == "__main__":
    sampler = SmartSampler()
    sampler.run()
