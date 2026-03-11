"""SFZ metadata generator"""
import os
import librosa
import soundfile as sf
import numpy as np


class SFZGenerator:
    """Generates SFZ metadata for samplers"""

    def _get_sample_frames(self, audio_path: str) -> int | None:
        """Get total number of frames in a WAV file."""
        try:
            info = sf.info(audio_path)
            return info.frames
        except Exception:
            return None
    
    def generate(self, audio_filename: str, detected_pitch: float | None, 
                 label: str, audio_path: str | None = None,
                 fixed_velocity: bool =False) -> str:
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

        loop_end_line = ""
        if audio_path is not None:
            frames = self._get_sample_frames(audio_path)
            if frames is not None and frames > 0:
                loop_end_line = f"loop_end={frames - 1}"

        if fixed_velocity:
            vel_lines = "lovel=0\nhivel=127\namp_veltrack=0"
        else:
            vel_lines = "lovel=0\nhivel=127"
        
        sfz_content = f"""// {label} Sample
{pitch_comment}
<region>
sample={audio_filename}
pitch_keycenter={int(round(midi_note))}
lokey=0
hikey=127
{vel_lines}

// Loop settings — sustain loop while key is held
loop_mode=loop_sustain
loop_start=0
{loop_end_line}
"""
        
        return sfz_content
    
    def save(self, output_path: str, audio_filename: str, 
             detected_pitch: float | None, label: str,
             fixed_velocity: bool = False):
        """
        Generate and save SFZ file.
        
        Args:
            output_path: Path to save SFZ file
            audio_filename: Name of the audio file (not full path)
            detected_pitch: Detected pitch in Hz
            label: Sample label/name
        """

        audio_dir  = os.path.dirname(output_path)
        audio_path = os.path.join(audio_dir, audio_filename)

        sfz_content = self.generate(audio_filename, detected_pitch, label, audio_path=audio_path, fixed_velocity=fixed_velocity)
        
        with open(output_path, 'w') as f:
            f.write(sfz_content)
        
        print(f"✅ SFZ metadata saved as {output_path}")