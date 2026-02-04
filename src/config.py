"""Configuration settings for Smart Sampler"""
import os

# === PATHS ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yamnet_model')
SAMPLES_DIR = os.path.join(BASE_DIR, 'samples')
TEST_SAMPLES_DIR = os.path.join(BASE_DIR, 'test_samples')

# === AUDIO SETTINGS ===
SAMPLE_RATE = 16000
RECORD_SECONDS = 6
CHANNELS = 1
CHUNK_SIZE = 1024

# === PROCESSING SETTINGS ===
TRIM_TOP_DB = 25  # dB threshold for silence trimming
NORMALIZE_PEAK = 0.99  # Peak amplitude after normalization

# === PITCH SETTINGS ===
PITCH_FMIN = 'C2'  # Lowest detectable pitch (~65 Hz)
PITCH_FMAX = 'C7'  # Highest detectable pitch (~2093 Hz)
TRANSPOSE_THRESHOLD = 0.5  # Semitones - skip if already close to target

# === TEST MODE ===
TEST_MODE = False  # Set to False when microphone is available
TEST_AUDIO_PATH = os.path.join(TEST_SAMPLES_DIR, 'test_input.wav')

# === FILENAMES ===
RAW_FILENAME = 'sample_raw.wav'
CLEAN_FILENAME = 'sample_clean.wav'
