"""Lightweight torchcodec compatibility stub for local audio pipelines.

The app feeds librosa-loaded or in-memory waveforms to transformers/pyannote, so
native torchcodec decoding is intentionally not used.
"""


class AudioDecoder:  # pylint: disable=too-few-public-methods
    """Placeholder matching torchcodec's public decoder name."""


class AudioSamples:  # pylint: disable=too-few-public-methods
    """Placeholder matching torchcodec's public sample container name."""


class AudioStreamMetadata:  # pylint: disable=too-few-public-methods
    """Placeholder matching torchcodec's public metadata name."""
