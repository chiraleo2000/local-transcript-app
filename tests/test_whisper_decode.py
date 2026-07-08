"""Unit tests for Whisper decode kwargs (repetition controls, no GPU needed)."""

from __future__ import annotations

import importlib

whisper_utils = importlib.import_module("engines.whisper_utils")


class TestRepetitionControls:
    def test_neutral_defaults_add_nothing(self, monkeypatch):
        monkeypatch.delenv("ASR_NO_REPEAT_NGRAM_SIZE", raising=False)
        monkeypatch.delenv("ASR_REPETITION_PENALTY", raising=False)
        kwargs = whisper_utils.whisper_generate_kwargs("thai")
        assert "no_repeat_ngram_size" not in kwargs
        assert "repetition_penalty" not in kwargs

    def test_ngram_applied_when_set(self, monkeypatch):
        monkeypatch.setenv("ASR_NO_REPEAT_NGRAM_SIZE", "6")
        kwargs = whisper_utils.whisper_generate_kwargs("thai")
        assert kwargs["no_repeat_ngram_size"] == 6

    def test_zero_ngram_disables(self, monkeypatch):
        monkeypatch.setenv("ASR_NO_REPEAT_NGRAM_SIZE", "0")
        kwargs = whisper_utils.whisper_generate_kwargs("thai")
        assert "no_repeat_ngram_size" not in kwargs

    def test_penalty_applied_when_above_one(self, monkeypatch):
        monkeypatch.setenv("ASR_REPETITION_PENALTY", "1.1")
        kwargs = whisper_utils.whisper_generate_kwargs("thai")
        assert kwargs["repetition_penalty"] == 1.1

    def test_penalty_of_one_disables(self, monkeypatch):
        monkeypatch.setenv("ASR_REPETITION_PENALTY", "1.0")
        kwargs = whisper_utils.whisper_generate_kwargs("thai")
        assert "repetition_penalty" not in kwargs


class TestChunkTimestampRepair:
    def test_fills_missing_end_from_next_chunk(self):
        result = {
            "text": "hello world",
            "chunks": [
                {"text": "hello", "timestamp": (0.0, None)},
                {"text": "world", "timestamp": (1.2, 2.0)},
            ],
        }
        repaired = whisper_utils.patch_missing_chunk_end_timestamps(result, 3.0)
        assert repaired["chunks"][0]["timestamp"] == (0.0, 1.2)

    def test_fills_missing_end_from_audio_duration(self):
        result = {
            "text": "tail",
            "chunks": [{"text": "tail", "timestamp": (4.0, None)}],
        }
        repaired = whisper_utils.patch_missing_chunk_end_timestamps(result, 5.0)
        assert repaired["chunks"][0]["timestamp"] == (4.0, 5.0)


class TestPipelineInitKwargs:
    def test_no_default_return_timestamps(self, monkeypatch):
        monkeypatch.delenv("ASR_PIPELINE_INTERNAL_CHUNK", raising=False)
        kwargs = whisper_utils.hf_pipeline_init_kwargs(lambda: 30)
        assert "return_timestamps" not in kwargs
        assert "chunk_length_s" not in kwargs

    def test_max_new_tokens_within_whisper_limit(self, monkeypatch):
        monkeypatch.delenv("ASR_PIPELINE_INTERNAL_CHUNK", raising=False)
        kwargs = whisper_utils.hf_pipeline_init_kwargs(lambda: 30)
        assert kwargs["max_new_tokens"] <= 444


class TestTurnLineTimestamps:
    def test_uses_diar_bounds_in_accuracy_mode(self, monkeypatch):
        from engines.whisper_runtime import _turn_line_timestamp_bounds

        monkeypatch.setenv("ASR_TURN_USE_DIAR_TIMESTAMPS", "true")
        monkeypatch.setenv("DIARIZATION_ACCURACY_MODE", "true")
        turn = {"start": 0.0, "end": 7.2, "speaker": "SPEAKER_01"}
        result = {"chunks": [{"text": "x", "timestamp": (0.0, 0.0)}]}
        start, end = _turn_line_timestamp_bounds(
            turn,
            result,
            slice_offset=0.0,
            turn_start=0.0,
            turn_end=7.2,
        )
        assert start == 0
        assert end == 7

    def test_word_bounds_fallback_when_collapsed(self, monkeypatch):
        from engines.whisper_runtime import _line_timestamp_bounds

        monkeypatch.setenv("ASR_WORD_TIMESTAMPS_WITH_DIARIZATION", "true")
        result = {"chunks": [{"text": "x", "timestamp": (0.0, 0.0)}]}
        start, end = _line_timestamp_bounds(
            result,
            slice_offset=0.0,
            turn_start=0.0,
            turn_end=7.0,
            fallback_start=0.0,
            fallback_end=7.0,
        )
        assert start == 0
        assert end == 7


class TestCudaErrorClassification:
    def test_device_not_ready_is_recoverable(self):
        from engines.whisper_runtime import is_cuda_recoverable, is_cuda_unknown_error

        exc = RuntimeError("CUDA driver error: device not ready")
        assert is_cuda_unknown_error(exc) is True
        assert is_cuda_recoverable(exc) is True


class TestHallucinationRejection:
    def test_rejects_repetitive_short_turn(self):
        from engines.whisper_runtime import _reject_hallucinated_turn

        text = "word " * 80
        assert _reject_hallucinated_turn(text, 1.0) is True

    def test_accepts_normal_short_turn(self):
        from engines.whisper_runtime import _reject_hallucinated_turn

        assert _reject_hallucinated_turn("สวัสดีครับ", 2.0) is False
        assert _reject_hallucinated_turn("ครับ ใช่ครับ", 0.4) is False

    def test_rejects_repeated_ngram(self):
        from engines.whisper_runtime import _reject_hallucinated_turn

        text = "one two three four " * 4
        assert _reject_hallucinated_turn(text, 8.0) is True
