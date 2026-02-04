"""Audio processors for Smart Sampler"""
from .recorder import AudioRecorder
from .trimmer import AudioTrimmer
from .normalizer import AudioNormalizer
from .pitch_detector import PitchDetector
from .transposer import PitchTransposer
from .classifier import AudioClassifier

__all__ = [
    'AudioRecorder',
    'AudioTrimmer', 
    'AudioNormalizer',
    'PitchDetector',
    'PitchTransposer',
    'AudioClassifier'
]
