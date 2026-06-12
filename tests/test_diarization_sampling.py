"""Tests for multi-pass diarization sampling."""

from engines.diarization_sampling import sample_param_sets, score_segments


def test_sample_param_sets_respects_limit(monkeypatch):
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE", "true")
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE_PASSES", "3")
    monkeypatch.delenv("DIARIZATION_ACCURACY_MODE", raising=False)
    configs = sample_param_sets(max_speakers=3, audio_duration_s=45.0)
    assert len(configs) == 3


def test_zero_passes_skips_grid_sweep(monkeypatch):
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE", "true")
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE_PASSES", "0")
    monkeypatch.setenv("DIARIZATION_ACCURACY_MODE", "true")
    configs = sample_param_sets(max_speakers=3, audio_duration_s=45.0)
    assert configs == []


def test_full_grid_when_explicit_flag(monkeypatch):
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE", "true")
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE_PASSES", "0")
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE_FULL_GRID", "true")
    monkeypatch.setenv("DIARIZATION_ACCURACY_MODE", "true")
    configs = sample_param_sets(max_speakers=3, audio_duration_s=45.0)
    assert len(configs) > 8


def test_score_segments_prefers_coverage():
    segments = [
        {"start": 0.0, "end": 5.0, "speaker": "A"},
        {"start": 5.0, "end": 10.0, "speaker": "B"},
    ]
    score = score_segments(segments, audio_duration_s=10.0, max_speakers=2)
    assert score > 0.5


def test_score_segments_rejects_single_speaker_collapse():
    segments = [{"start": 0.0, "end": 200.0, "speaker": "A"}]
    score = score_segments(segments, audio_duration_s=200.0, max_speakers=3)
    assert score < 0


def test_select_best_diarization_params_returns_winner_params(monkeypatch):
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE", "true")
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE_PASSES", "2")
    monkeypatch.setenv("DIARIZATION_ACCURACY_MODE", "true")

    calls: list[str] = []

    def instantiate(params, label):
        calls.append(label)

    def run_fn():
        if len(calls) <= 1:
            return [{"start": 0.0, "end": 10.0, "speaker": "A"}]
        return [
            {"start": 0.0, "end": 5.0, "speaker": "A"},
            {"start": 5.0, "end": 10.0, "speaker": "B"},
        ]

    from engines.diarization_sampling import select_best_diarization_params

    params, label, score = select_best_diarization_params(
        instantiate,
        run_fn,
        audio_duration_s=10.0,
        max_speakers=2,
        base_params={"segmentation": {"threshold": 0.40}, "clustering": {"threshold": 0.48}},
    )
    assert params is not None
    assert label != "none"
    assert score > 0


def test_score_segments_penalizes_mega_turn(monkeypatch):
    monkeypatch.setenv("DIARIZATION_ACCURACY_MODE", "true")
    segments = [
        {"start": 0.0, "end": 180.0, "speaker": "A"},
        {"start": 180.0, "end": 200.0, "speaker": "B"},
    ]
    bad = score_segments(segments, audio_duration_s=200.0, max_speakers=3)
    good = score_segments([
        {"start": 0.0, "end": 60.0, "speaker": "A"},
        {"start": 60.0, "end": 120.0, "speaker": "B"},
        {"start": 120.0, "end": 200.0, "speaker": "C"},
    ], audio_duration_s=200.0, max_speakers=3)
    assert good > bad

