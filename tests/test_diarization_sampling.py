"""Tests for multi-pass diarization sampling."""

from engines.diarization_sampling import (
    effective_pass_limit,
    sample_param_sets,
    score_segments,
    tune_window_bounds,
)


def test_sample_param_sets_respects_limit(monkeypatch):
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE", "true")
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE_PASSES", "3")
    monkeypatch.delenv("DIARIZATION_ACCURACY_MODE", raising=False)
    configs = sample_param_sets(max_speakers=3, audio_duration_s=45.0)
    assert len(configs) == 3


def test_curated_configs_prioritized_in_accuracy_mode(monkeypatch):
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE", "true")
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE_PASSES", "4")
    monkeypatch.setenv("DIARIZATION_ACCURACY_MODE", "true")
    configs = sample_param_sets(max_speakers=2, audio_duration_s=120.0)
    assert len(configs) == 4
    assert configs[0][0].startswith("curated_")


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


def test_effective_pass_limit_caps_8gb(monkeypatch):
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE", "true")
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE_PASSES", "6")
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE_PASSES_8GB", "4")
    monkeypatch.setattr(
        "engines.diarization_sampling._strict_8gb_class",
        lambda: True,
    )
    assert effective_pass_limit(120.0, 3) == 4


def test_tune_window_bounds_for_long_audio(monkeypatch):
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE_TUNE_WINDOW", "true")
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE_TUNE_MIN_AUDIO_S", "300")
    bounds = tune_window_bounds(600.0)
    assert bounds is not None
    start, end = bounds
    assert end - start >= 60.0
    assert start >= 0.0


def test_tune_window_skipped_for_short_audio(monkeypatch):
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE_TUNE_WINDOW", "true")
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE_TUNE_MIN_AUDIO_S", "300")
    assert tune_window_bounds(120.0) is None


def test_tune_window_skipped_when_multi_sr_enabled(monkeypatch):
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE_TUNE_WINDOW", "true")
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE_TUNE_MIN_AUDIO_S", "300")
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE_SR", "16000,44100")
    assert tune_window_bounds(600.0) is None


def test_multi_sample_max_tries_caps_total_passes(monkeypatch):
    from engines.diarization_sampling import (
        multi_sample_max_total_tries,
        multi_sample_max_tries,
    )

    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE_MAX_TOTAL", "9")
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE_SR", "16000,22050,44100")
    monkeypatch.delenv("DIARIZATION_MULTI_SAMPLE_MAX_TRIES", raising=False)
    assert multi_sample_max_total_tries() == 9
    assert multi_sample_max_tries() == 3


def test_run_multi_sample_respects_max_tries(monkeypatch):
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE", "true")
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE_PASSES", "8")
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE_MAX_TRIES", "3")
    monkeypatch.setenv("DIARIZATION_ACCURACY_MODE", "true")
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE_EARLY_STOP", "false")

    calls: list[str] = []

    def instantiate(params, label):
        calls.append(label)

    def run_fn():
        return [{"start": 0.0, "end": 5.0, "speaker": "A"}]

    from engines.diarization_sampling import run_multi_sample_diarization

    run_multi_sample_diarization(
        instantiate,
        run_fn,
        audio_duration_s=30.0,
        max_speakers=2,
        base_params={"segmentation": {"threshold": 0.40}},
    )
    assert len(calls) == 3


def test_multi_sample_preprocess_srs_parses_csv(monkeypatch):
    from engines.diarization_sampling import multi_sample_preprocess_srs

    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE_SR", "16000,22050,44100")
    assert multi_sample_preprocess_srs() == [16000, 22050, 44100]
    monkeypatch.delenv("DIARIZATION_MULTI_SAMPLE_SR", raising=False)
    assert multi_sample_preprocess_srs() == []


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


def test_early_stop_after_good_score(monkeypatch):
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE", "true")
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE_PASSES", "6")
    monkeypatch.setenv("DIARIZATION_ACCURACY_MODE", "true")
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE_EARLY_STOP", "true")
    monkeypatch.setenv("DIARIZATION_MULTI_SAMPLE_EARLY_STOP_SCORE", "0.70")

    calls: list[str] = []

    def instantiate(params, label):
        calls.append(label)

    def run_fn():
        return [
            {"start": 0.0, "end": 5.0, "speaker": "A"},
            {"start": 5.0, "end": 10.0, "speaker": "B"},
        ]

    from engines.diarization_sampling import run_multi_sample_diarization

    _segments, _label, _score = run_multi_sample_diarization(
        instantiate,
        run_fn,
        audio_duration_s=10.0,
        max_speakers=2,
        base_params=None,
    )
    assert len(calls) < 6


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
