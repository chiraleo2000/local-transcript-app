"""Lightweight torchcodec compatibility stub for local audio pipelines.

The app feeds librosa-loaded or in-memory waveforms to transformers/pyannote, so
native torchcodec decoding is intentionally not used.
"""

from . import decoders, encoders, samplers, transforms
from .decoders import AudioDecoder, AudioSamples, AudioStreamMetadata

__all__ = [
    "AudioDecoder",
    "AudioSamples",
    "AudioStreamMetadata",
    "decoders",
    "encoders",
    "samplers",
    "transforms",
]
