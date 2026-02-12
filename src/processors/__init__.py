"""Audio processors for Smart Sampler"""
from .recorder import AudioRecorder
from .trimmer import AudioTrimmer
from .normalizer import AudioNormalizer
from .pitch_detector import PitchDetector
from .transposer import PitchTransposer
from .classifier import AudioClassifier
from .high_pass_filter import HighPassFilter
from .dtln_denoiser import DTLNDenoiser

__all__ = [
    'AudioRecorder',
    'AudioTrimmer', 
    'AudioNormalizer',
    'PitchDetector',
    'PitchTransposer',
    'AudioClassifier',
    'HighPassFilter',
    'DTLNDenoiser'
]
