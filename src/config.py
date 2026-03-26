"""Configuration settings for Smart Sampler"""
import os

# === PATHS ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yamnet_model')
_USB_MOUNT  = '/media/fintanpi/INTENSO'
_USB_PATH   = _USB_MOUNT + '/smart_sampler_samples'
SAMPLES_DIR = _USB_PATH if os.path.ismount(_USB_MOUNT) else os.path.join(BASE_DIR, 'samples')
IMPORT_DIR  = (_USB_MOUNT + '/imported_samples') if os.path.ismount(_USB_MOUNT) else os.path.join(BASE_DIR, 'imported_samples')
TEST_SAMPLES_DIR = IMPORT_DIR

# === AUDIO SETTINGS ===
SAMPLE_RATE = 16000
TASCAM_SAMPLE_RATE = 48000
RECORD_SECONDS = 6
CHANNELS = 2
CHUNK_SIZE = 1024

# === PROCESSING SETTINGS ===
TRIM_TOP_DB = 25  # dB threshold for silence trimming
NORMALIZE_PEAK = 0.99  # Peak amplitude after normalization

# === FILTER SETTINGS ===
HPF_CUTOFF = 80  # High pass filter cutoff frequency in Hz
HPF_ORDER = 4    # Filter order (4 = 24dB/octave rolloff)

# === DTLN SETTINGS ===
DTLN_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'dtln')
DTLN_BLOCK_LEN = 512
DTLN_BLOCK_SHIFT = 128

# === PITCH SETTINGS ===
PITCH_FMIN = 'C2'  # Lowest detectable pitch (~65 Hz)
PITCH_FMAX = 'C7'  # Highest detectable pitch (~2093 Hz)
TRANSPOSE_THRESHOLD = 0.5  # Semitones - skip if already close to target

# === TEST MODE ===
TEST_MODE = False  # Set to False when microphone is available
TEST_AUDIO_PATH = os.path.join(IMPORT_DIR, 'placeholder.wav')

# === FILENAMES ===
RAW_FILENAME = 'sample_raw.wav'
CLEAN_FILENAME = 'sample_clean.wav'

# === MIDI PLAYER SETTINGS ===
SFIZZ_BINARY = 'sfizz_jack'
JACK_AUDIO_DEVICE = 'hw:3'
JACK_SAMPLE_RATE  = 44100
JACK_BUFFER_SIZE  = 256