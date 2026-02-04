# Smart Sampler

Audio processing pipeline for Raspberry Pi that records, classifies, and processes audio samples for music production.

## Features

- **Audio Recording**: Record samples from microphone
- **Silence Trimming**: Automatically trim silence from beginning and end
- **Normalization**: Peak normalization for consistent levels
- **Pitch Detection**: Detect fundamental frequency using pYIN algorithm
- **Pitch Transposition**: Transpose to nearest C note
- **Audio Classification**: Classify samples using Google's YAMNet model
- **Visualization**: Generate spectrograms for comparison
- **File Organization**: Automatically organize samples by classification label

## Setup

Run the setup script to install all dependencies and download the YAMNet model:

```bash
chmod +x setup_project.sh
./setup_project.sh
```

This will:
1. Install system dependencies (portaudio, etc.)
2. Create a virtual environment
3. Install Python packages
4. Download the YAMNet model
5. Set up directory structure

## Usage

### Live Mode (with microphone)

1. Edit `src/config.py` and set `TEST_MODE = False`
2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```
3. Run the sampler:
   ```bash
   python src/main.py
   ```

### Test Mode (without microphone)

1. Place a WAV file in `test_samples/test_input.wav`
2. Ensure `TEST_MODE = True` in `src/config.py` (default)
3. Run the sampler:
   ```bash
   source venv/bin/activate
   python src/main.py
   ```

## Configuration

Edit `src/config.py` to customize:
- Sample rate
- Recording duration
- Trim threshold
- Normalization peak
- Pitch detection range
- Test mode settings

## Project Structure

```
smart_sampler_master/
├── src/
│   ├── main.py              # Main entry point
│   ├── config.py            # Configuration settings
│   ├── processors/          # Audio processing modules
│   │   ├── recorder.py
│   │   ├── trimmer.py
│   │   ├── normalizer.py
│   │   ├── pitch_detector.py
│   │   ├── transposer.py
│   │   └── classifier.py
│   └── utils/               # Utility modules
│       ├── audio_io.py
│       ├── visualization.py
│       └── file_manager.py
├── models/
│   └── yamnet_model/        # YAMNet classification model
├── samples/                 # Organized output samples
├── test_samples/            # Test input files
├── requirements.txt
├── setup.py
└── setup_project.sh
```

## Output

Processed samples are saved in `samples/[label]/` with the following naming:
- `[label]_[timestamp]_CLEAN_[note].wav` - Processed audio
- `[label]_[timestamp]_RAW.wav` - Original recording
- `[label]_[timestamp]_SPECTROGRAM.png` - Visualization

## Requirements

- Python 3.9+
- Raspberry Pi OS (or any Debian-based Linux)
- Microphone (for live mode)
