"""Audio enhancement defaults and filter chain."""

from engines.preprocess import _ffmpeg_filter_chain


def test_ffmpeg_filter_includes_atempo_when_configured(monkeypatch):
    monkeypatch.setenv("AUDIO_ENHANCE_ATEMPO", "0.92")
    monkeypatch.setenv("AUDIO_ENHANCE_LOUDNORM_I", "-14")
    chain = _ffmpeg_filter_chain()
    assert "atempo=0.920" in chain
    assert "loudnorm=I=-14" in chain


def test_ffmpeg_filter_skips_atempo_at_unity(monkeypatch):
    monkeypatch.setenv("AUDIO_ENHANCE_ATEMPO", "1.0")
    chain = _ffmpeg_filter_chain()
    assert "atempo" not in chain
