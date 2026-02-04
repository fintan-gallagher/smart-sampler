"""Audio recording module with USB device support"""
import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
from ..config import SAMPLE_RATE, CHANNELS


class AudioRecorder:
    """Records audio from input device"""
    
    # Tascam native sample rate (48kHz)
    TASCAM_SAMPLE_RATE = 48000
    
    def find_tascam_device(self):
        """Find Tascam DR-05X USB audio device"""
        devices = sd.query_devices()
        for idx, device in enumerate(devices):
            name_lower = device['name'].lower()
            if 'tascam' in name_lower or 'dr series' in name_lower or 'dr-05' in name_lower:
                if device['max_input_channels'] > 0:
                    print(f"✅ Found Tascam: {device['name']}")
                    return idx
        return None
    
    def record(self, output_path: str):
        """
        Record audio continuously until user stops.
        
        Args:
            output_path: Path to save recording
        """
        # Find Tascam device
        device_id = self.find_tascam_device()
        
        if device_id is None:
            print("❌ Tascam DR-05X not found!")
            print("\nAvailable input devices:")
            for idx, device in enumerate(sd.query_devices()):
                if device['max_input_channels'] > 0:
                    print(f"   [{idx}] {device['name']}")
            
            # Fallback to default device
            print("\nUsing default input device instead...")
            device_id = None
        
        print(f"\n🎙️ Recording from Tascam DR-05X...")
        print(f"   Native sample rate: {self.TASCAM_SAMPLE_RATE} Hz")
        print(f"   Target sample rate: {SAMPLE_RATE} Hz")
        print("   Press ENTER to stop recording\n")
        
        # Record in chunks at Tascam's native sample rate
        recording = []
        
        def callback(indata, frames, time, status):
            if status:
                print(f"Recording status: {status}")
            recording.append(indata.copy())
        
        try:
            with sd.InputStream(
                device=device_id,
                channels=CHANNELS,
                samplerate=self.TASCAM_SAMPLE_RATE,
                callback=callback
            ):
                input("   [Recording... Press ENTER to stop]")
        except KeyboardInterrupt:
            print("\n⚠️ Recording interrupted")
        except Exception as e:
            print(f"❌ Recording error: {e}")
            return
        
        # Concatenate all chunks
        if recording:
            audio = np.concatenate(recording, axis=0)
            
            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample to target sample rate (16kHz for YAMNet)
            if self.TASCAM_SAMPLE_RATE != SAMPLE_RATE:
                print(f"   Resampling {self.TASCAM_SAMPLE_RATE} Hz → {SAMPLE_RATE} Hz...")
                audio = librosa.resample(
                    audio, 
                    orig_sr=self.TASCAM_SAMPLE_RATE, 
                    target_sr=SAMPLE_RATE
                )
            
            # Save at target sample rate
            sf.write(output_path, audio, SAMPLE_RATE)
            print(f"✅ Recording saved: {output_path}")
            print(f"   Duration: {len(audio)/SAMPLE_RATE:.2f}s")
        else:
            print("❌ No audio recorded")