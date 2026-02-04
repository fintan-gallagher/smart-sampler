#!/bin/bash

echo "========================================"
echo "🎵 Smart Sampler Setup Script"
echo "========================================"

# Exit on error
set -e

# Get project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo ""
echo "📁 Project directory: $PROJECT_DIR"

# === Step 1: Install system dependencies ===
echo ""
echo "📦 Step 1: Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-full \
    python3-venv \
    python3-dev \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg

# === Step 2: Create virtual environment ===
echo ""
echo "🐍 Step 2: Creating virtual environment..."
if [ -d "venv" ]; then
    echo "   Virtual environment already exists, skipping..."
else
    python3 -m venv venv
    echo "   ✅ Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate
echo "   ✅ Virtual environment activated"

# === Step 3: Upgrade pip ===
echo ""
echo "⬆️  Step 3: Upgrading pip..."
pip install --upgrade pip wheel setuptools

# === Step 4: Install Python dependencies ===
echo ""
echo "📚 Step 4: Installing Python dependencies..."
pip install -r requirements.txt

# === Step 5: Download YAMNet model ===
echo ""
echo "🤖 Step 5: Downloading YAMNet model..."

YAMNET_DIR="$PROJECT_DIR/models/yamnet_model"
YAMNET_URL="https://tfhub.dev/google/yamnet/1?tf-hub-format=compressed"

if [ -d "$YAMNET_DIR" ] && [ -f "$YAMNET_DIR/saved_model.pb" ]; then
    echo "   YAMNet model already exists, skipping..."
else
    echo "   Downloading YAMNet..."
    mkdir -p "$YAMNET_DIR"
    
    # Download and extract
    TMP_FILE="/tmp/yamnet.tar.gz"
    wget -q --show-progress -O "$TMP_FILE" "$YAMNET_URL"
    
    echo "   Extracting model..."
    tar -xzf "$TMP_FILE" -C "$YAMNET_DIR"
    rm "$TMP_FILE"
    
    echo "   ✅ YAMNet model downloaded"
fi

# === Step 6: Download YAMNet class map ===
echo ""
echo "📋 Step 6: Downloading YAMNet class map..."

CLASS_MAP_DIR="$YAMNET_DIR/assets"
CLASS_MAP_FILE="$CLASS_MAP_DIR/yamnet_class_map.csv"
CLASS_MAP_URL="https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"

if [ -f "$CLASS_MAP_FILE" ]; then
    echo "   Class map already exists, skipping..."
else
    mkdir -p "$CLASS_MAP_DIR"
    wget -q --show-progress -O "$CLASS_MAP_FILE" "$CLASS_MAP_URL"
    echo "   ✅ Class map downloaded"
fi

# === Step 7: Create directory structure ===
echo ""
echo "📂 Step 7: Creating directory structure..."
mkdir -p samples
mkdir -p test_samples
mkdir -p models

# === Step 8: Verify installation ===
echo ""
echo "🔍 Step 8: Verifying installation..."

python3 << 'EOF'
import sys
print(f"   Python: {sys.version}")

try:
    import numpy as np
    print(f"   ✅ NumPy {np.__version__}")
except ImportError as e:
    print(f"   ❌ NumPy: {e}")

try:
    import librosa
    print(f"   ✅ Librosa {librosa.__version__}")
except ImportError as e:
    print(f"   ❌ Librosa: {e}")

try:
    import tensorflow as tf
    print(f"   ✅ TensorFlow {tf.__version__}")
except ImportError as e:
    print(f"   ❌ TensorFlow: {e}")

try:
    import soundfile as sf
    print(f"   ✅ SoundFile {sf.__version__}")
except ImportError as e:
    print(f"   ❌ SoundFile: {e}")

try:
    import pyaudio
    print(f"   ✅ PyAudio {pyaudio.__version__}")
except ImportError as e:
    print(f"   ❌ PyAudio: {e}")

try:
    import matplotlib
    print(f"   ✅ Matplotlib {matplotlib.__version__}")
except ImportError as e:
    print(f"   ❌ Matplotlib: {e}")

try:
    import scipy
    print(f"   ✅ SciPy {scipy.__version__}")
except ImportError as e:
    print(f"   ❌ SciPy: {e}")
EOF

# === Step 9: Test YAMNet loading ===
echo ""
echo "🧪 Step 9: Testing YAMNet model loading..."

python3 << EOF
import os
import tensorflow as tf

model_path = "$YAMNET_DIR"
try:
    model = tf.saved_model.load(model_path)
    print("   ✅ YAMNet model loads successfully!")
except Exception as e:
    print(f"   ❌ YAMNet model failed to load: {e}")
EOF

echo ""
echo "========================================"
echo "✅ Setup complete!"
echo "========================================"
echo ""
echo "To activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "To run the sampler:"
echo "   python src/main.py"
echo ""
echo "To run in test mode (no microphone):"
echo "   Place a WAV file in test_samples/test_input.wav"
echo ""
