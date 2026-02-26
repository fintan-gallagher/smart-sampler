"""Entry point — run this to launch the Smart Sampler UI"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# PipeWire JACK latency — must be set before any audio initialisation
os.environ.setdefault('PIPEWIRE_LATENCY', '256/44100')

# Use PulseAudio compat
os.environ['SDL_AUDIODRIVER'] = 'pulse'
os.environ['PULSE_SERVER']    = f'unix:/run/user/{os.getuid()}/pulse/native'

# Pi framebuffer — ignored when DISPLAY is set (desktop)
if not os.environ.get('DISPLAY'):
    os.environ.setdefault('SDL_VIDEODRIVER', 'fbcon')
    os.environ.setdefault('SDL_FBDEV',       '/dev/fb0')

from src.ui import SamplerApp

if __name__ == '__main__':
    app = SamplerApp()
    app.run()