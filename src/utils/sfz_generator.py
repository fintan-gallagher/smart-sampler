"""SFZ metadata generator"""
import os
import librosa


class SFZGenerator:
    """Generates SFZ metadata for samplers"""
    
    def generate(self, audio_filename: str, detected_pitch: float | None, 
                 label: str) -> str:
        """
        Generate SFZ metadata for a sample.
        
        Args:
            audio_filename: Name of the audio file (not full path)
            detected_pitch: Detected pitch in Hz
            label: Sample label/name
            
        Returns:
            SFZ content as string
        """
        # Set defaults first
        midi_note = 60
        note_name = "C4"
        pitch_comment = "// No pitch detected - defaulting to C4"
        
        # Override if pitch was detected
        if detected_pitch is not None:
            midi_note = librosa.hz_to_midi(detected_pitch)
            note_name = librosa.hz_to_note(detected_pitch)
            pitch_comment = f"// Detected Pitch: {detected_pitch:.2f} Hz ({note_name})"
        
        sfz_content = f"""// {label} Sample
{pitch_comment}
<region>
sample={audio_filename}
pitch_keycenter={int(round(midi_note))}
lokey=0
hikey=127
lovel=0
hivel=127
"""
        
        return sfz_content
    
    def save(self, output_path: str, audio_filename: str, 
             detected_pitch: float | None, label: str):
        """
        Generate and save SFZ file.
        
        Args:
            output_path: Path to save SFZ file
            audio_filename: Name of the audio file (not full path)
            detected_pitch: Detected pitch in Hz
            label: Sample label/name
        """
        sfz_content = self.generate(audio_filename, detected_pitch, label)
        
        with open(output_path, 'w') as f:
            f.write(sfz_content)
        
        print(f"✅ SFZ metadata saved as {output_path}")