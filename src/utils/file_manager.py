"""File management utilities"""
import os
import shutil
from datetime import datetime
import librosa
from ..config import SAMPLES_DIR


class FileManager:
    """Handles file organization and saving"""
    
    def __init__(self, samples_dir: str = SAMPLES_DIR):
        self.samples_dir = samples_dir
        os.makedirs(samples_dir, exist_ok=True)
    
    def save_with_label(self, clean_audio_path: str, raw_audio_path: str,
                        label: str, target_pitch: float | None,
                        spectrogram_func=None, sfz_func=None) -> dict:
        """
        Save files organized by label.
        
        Args:
            clean_audio_path: Path to processed audio
            raw_audio_path: Path to raw audio
            label: Classification label
            target_pitch: Target pitch in Hz (or None)
            spectrogram_func: Optional function to generate spectrogram
            sfz_func: Optional function to generate SFZ metadata
            
        Returns:
            Dict with saved file paths
        """
        # Create label folder
        label_folder = os.path.join(self.samples_dir, label)
        os.makedirs(label_folder, exist_ok=True)
        
        # Generate timestamped filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if target_pitch:
            note_name = librosa.hz_to_note(target_pitch).replace('#', 'sharp')
            clean_name = f"{label}_{timestamp}_CLEAN_{note_name}.wav"
        else:
            clean_name = f"{label}_{timestamp}_CLEAN_UNPITCHED.wav"
        
        raw_name = f"{label}_{timestamp}_RAW.wav"
        spectrogram_name = f"{label}_{timestamp}_SPECTROGRAM.png"
        sfz_name = f"{label}_{timestamp}.sfz"
        
        # Build paths
        paths = {
            'raw': os.path.join(label_folder, raw_name),
            'clean': os.path.join(label_folder, clean_name),
            'spectrogram': os.path.join(label_folder, spectrogram_name),
            'sfz': os.path.join(label_folder, sfz_name)
        }
        
        # Copy/move files
        shutil.copy2(raw_audio_path, paths['raw'])
        shutil.move(clean_audio_path, paths['clean'])
        
        # Generate spectrogram if function provided
        if spectrogram_func:
            spectrogram_func(paths['spectrogram'])

        #Generate SFZ if function provided
        if sfz_func:
            sfz_func(paths['sfz'], clean_name)
        
        return paths
    
    def prompt_label_selection(self, predictions: list[tuple[str, float]]) -> str:
        """
        Prompt user to select label from predictions.
        
        Args:
            predictions: List of (label, confidence) tuples
            
        Returns:
            Selected label string
        """
        print(f"\n🏷️ Top 3 predictions:")
        for i, (name, score) in enumerate(predictions, 1):
            print(f"   {i}. {name} (confidence: {score:.3f})")
        
        print(f"\n❓ Which label should be used?")
        print(f"   Enter 1, 2, or 3 to select")
        print(f"   Or press Enter to use the top prediction ({predictions[0][0]})")
        
        while True:
            choice = input("\nYour choice: ").strip()
            
            if choice == "":
                label = predictions[0][0]
                print(f"✅ Using default: {label}")
                return label
            elif choice in ["1", "2", "3"]:
                label = predictions[int(choice) - 1][0]
                print(f"✅ Selected: {label}")
                return label
            else:
                print("❌ Invalid input. Please enter 1, 2, 3, or press Enter.")
