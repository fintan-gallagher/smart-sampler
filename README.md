# Smart Sampler

Smart Sampler is a Raspberry Pi–friendly audio sampling pipeline + touchscreen UI.

It can:
- record audio (tested with a Tascam DR-series USB device, falls back to the system default input)
- apply high-pass filtering + optional DTLN noise reduction
- trim silence + normalize
- detect pitch
- classify sounds with Google's YAMNet model
- save organized samples plus spectrograms and SFZ files

This repo includes both:
- a **CLI pipeline** (terminal prompts)
- a **PyGame UI** designed for 480×320 screens (works windowed on desktop)

## Recommended Raspberry Pi setup

- Raspberry Pi 4/5 (4GB+ recommended)
- Raspberry Pi OS **64-bit** (Bookworm recommended)
- Python 3.9+ (Bookworm ships with 3.11)
- For the UI: a desktop session (X11/Wayland) is simplest. Headless console mode is also supported.

## Quick start (Raspberry Pi OS)

1) Clone the repo and run the setup script:

```bash
git clone <your-repo-url>
cd smart_sampler_master

chmod +x setup_project.sh
./setup_project.sh
```

What the script does:
- installs system packages (PortAudio, libsndfile, ffmpeg)
- creates `venv/`
- installs Python dependencies from `requirements.txt`
- downloads the YAMNet SavedModel into `models/yamnet_model/` (skips if already present)
- does **not** download DTLN: you download the DTLN TFLite files separately (instructions below)

2) Activate the venv:

```bash
source venv/bin/activate
```

3) Run either the UI or the CLI.

## Run: UI (recommended on Raspberry Pi)

The UI entrypoint is `run_ui.py`.

```bash
source venv/bin/activate
python run_ui.py
```

Notes:
- `run_ui.py` sets a few environment variables for PipeWire/PulseAudio and framebuffer usage.
- If you’re on the Pi console (no `DISPLAY` set), it will try to use the framebuffer (`SDL_VIDEODRIVER=fbcon`).

## Run: CLI pipeline

The CLI pipeline is a Python package module and should be run with `-m`:

```bash
source venv/bin/activate
python -m src.main
```

It will:
- in live mode: record from mic until you press Enter
- in test mode: list available WAV files to pick from

## Live mode vs test mode

Open `src/config.py` and set:
- `TEST_MODE = False` for recording from microphone
- `TEST_MODE = True` to process existing WAV files

### Test mode input folder

In test mode, the app looks in `IMPORT_DIR` (see `src/config.py`) and will list all `*.wav` files it finds.

By default, `IMPORT_DIR` points to:
- a USB drive path if the configured mount exists, otherwise
- the local repo folder `imported_samples/`

So the simplest workflow is:

```bash
mkdir -p imported_samples
cp your_audio.wav imported_samples/
python -m src.main
```

## USB drive workflow (optional)

`src/config.py` is set up to prefer a USB drive when mounted.

If you want samples to go to a USB stick/SSD, edit these in `src/config.py`:
- `_USB_MOUNT` (currently `/media/fintanpi/INTENSO`)

When the mount exists:
- processed output is written to `SAMPLES_DIR` on the USB drive
- test/imported WAVs are read from `IMPORT_DIR` on the USB drive

## Output files

Processed samples are saved under `SAMPLES_DIR/<label>/` (local `samples/` by default), typically including:
- `*_RAW.wav` (original)
- `*_CLEAN_*.wav` (processed)
- `*_SPECTROGRAM.png`
- `*.sfz`

## Models (YAMNet + DTLN)

This project uses two ML model bundles:

- **YAMNet (classification)**
    - Installed by [setup_project.sh](setup_project.sh) into `models/yamnet_model/`.
- **DTLN (optional denoising)**
    - Not included in this repo (the files are large and `*.tflite` is ignored).
    - You must download `model_1.tflite` and `model_2.tflite` into `models/dtln/`.

### Download DTLN models

The models come from the upstream DTLN project (MIT licensed): https://github.com/breizhn/DTLN

Run this from the repo root:

```bash
DTLN_COMMIT='1de1f15a8b5b7e1c44905618ff2ef70ca8277fbc'

mkdir -p models/dtln

curl -L -o models/dtln/model_1.tflite \
    "https://raw.githubusercontent.com/breizhn/DTLN/${DTLN_COMMIT}/pretrained_model/model_1.tflite"

curl -L -o models/dtln/model_2.tflite \
    "https://raw.githubusercontent.com/breizhn/DTLN/${DTLN_COMMIT}/pretrained_model/model_2.tflite"
```

Optional: verify file hashes:

```bash
cat <<'EOF' | sha256sum -c -
91281a38e80fe9fd330e28eda7e16fe4e483ee5199a3e687a099939013c25de0  models/dtln/model_1.tflite
7ae37ec802862d8a65b5cdabfbcbbe22caaf7cd39e79adf574d15837d1520830  models/dtln/model_2.tflite
EOF
```

## Troubleshooting (Raspberry Pi)

### Audio device permissions / no input device

If recording fails or you see “No Default Input Device”, try:

```bash
sudo usermod -aG audio $USER
reboot
```

You can also list input devices:

```bash
source venv/bin/activate
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### PyAudio build errors

If `pip` fails building `pyaudio`, ensure PortAudio headers are installed:

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-dev
```

Then re-run:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Pygame UI doesn’t open on the Pi console

- Best path: run from the desktop environment.
- For console mode: make sure you are on a local TTY (not SSH) and that framebuffer access works.

### TensorFlow install is slow / heavy

TensorFlow is the largest dependency (needed for YAMNet and DTLN). On Pi, first-time installs can take a while.

Tips:
- Use Raspberry Pi OS 64-bit.
- Keep swap enabled if you have low RAM.
- If `pip` is slow, Raspberry Pi OS commonly uses `piwheels` automatically.

### DTLN models missing

If DTLN denoising errors with “file not found”, confirm these exist:

- `models/dtln/model_1.tflite`
- `models/dtln/model_2.tflite`

If they’re missing, follow the download steps in “Download DTLN models” above.

## Project structure (high level)

```
smart_sampler_master/
├── run_ui.py                # UI entrypoint (PyGame)
├── setup_project.sh         # Pi-friendly setup
├── requirements.txt
├── models/
│   ├── dtln/                # DTLN TFLite models
│   └── yamnet_model/        # YAMNet SavedModel + class map
├── imported_samples/        # Drop WAVs here (when not using USB)
├── samples/                 # Output (when not using USB)
└── src/
    ├── config.py
    ├── main.py              # CLI pipeline module (run with: python -m src.main)
    ├── processors/
    ├── ui/
    └── utils/
```

## Running tests (optional)

```bash
source venv/bin/activate
pytest
```
