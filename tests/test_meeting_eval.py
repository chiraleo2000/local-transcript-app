"""Unit tests for named-reference meeting evaluation and VBx meeting tuning."""

from __future__ import annotations

import importlib

meeting_eval = importlib.import_module("tests.golden.meeting_eval")
diarization = importlib.import_module("engines.diarization")
sampling = importlib.import_module("engines.diarization_sampling")


class TestParseNamedReference:
    def test_parses_turns_with_implied_ends(self):
        text = (
            "header line without timestamp\n"
            "00:00:00 SpeakerA\n"
            "hello\n"
            "00:00:10 SpeakerB\n"
            "world\n"
        )
        turns = meeting_eval.parse_named_reference(text, total_duration_s=30.0)
        assert [t["speaker"] for t in turns] == ["SpeakerA", "SpeakerB"]
        assert turns[0]["end"] == 10.0
        assert turns[1]["end"] == 30.0

    def test_canonicalizes_typo_and_aliases(self):
        assert (
            meeting_eval.canonical_speaker_name("เกยรติศักดิ์ มาศกุล")
            == "เกียรติศักดิ์ มาศกุล"
        )
        assert (
            meeting_eval.canonical_speaker_name("เจ้าหน้าที่ สพก.5 (อนุพงษ์)")
            == "อนุพงษ์"
        )
        assert (
            meeting_eval.canonical_speaker_name("เจ้าหน้าที่ ศรชล. (สพก.5)")
            == "อนุพงษ์"
        )

    def test_reference_309_has_eleven_speakers(self):
        from pathlib import Path

        path = Path(__file__).parent / "309.txt"
        turns = meeting_eval.load_reference_turns(path, total_duration_s=5380.6)
        speakers = {t["speaker"] for t in turns}
        assert len(turns) == 635
        assert len(speakers) == 11


class TestEvaluateMeetingDiarization:
    def test_perfect_hypothesis_scores_high(self):
        ref = [
            {"start": 0.0, "end": 10.0, "speaker": "A"},
            {"start": 10.0, "end": 20.0, "speaker": "B"},
            {"start": 20.0, "end": 30.0, "speaker": "A"},
        ]
        hyp = [
            {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_01"},
            {"start": 10.0, "end": 20.0, "speaker": "SPEAKER_02"},
            {"start": 20.0, "end": 30.0, "speaker": "SPEAKER_01"},
        ]
        report = meeting_eval.evaluate_meeting_diarization(ref, hyp)
        assert report["detected_speakers"] == 2
        assert report["expected_speakers"] == 2
        assert report["speaker_count_match"] is True
        assert report["speaker_time_accuracy"] == 1.0
        assert report["turn_accuracy"] == 1.0
        assert report["boundary_within_1s"] == 1.0

    def test_merged_speakers_lower_attribution(self):
        ref = [
            {"start": 0.0, "end": 10.0, "speaker": "A"},
            {"start": 10.0, "end": 20.0, "speaker": "B"},
        ]
        hyp = [{"start": 0.0, "end": 20.0, "speaker": "SPEAKER_01"}]
        report = meeting_eval.evaluate_meeting_diarization(ref, hyp)
        assert report["detected_speakers"] == 1
        assert report["speaker_time_accuracy"] <= 0.55


class TestTurnTextAccuracy:
    def test_scores_matching_utterance_text(self):
        ref = [
            {"start": 0.0, "end": 5.0, "speaker": "A", "text": "hello world"},
            {"start": 5.0, "end": 10.0, "speaker": "B", "text": "goodbye"},
        ]
        hyp = [
            {
                "start": 0.0,
                "end": 5.0,
                "speaker": "SPEAKER_01",
                "text": "hello world",
            },
            {
                "start": 5.0,
                "end": 10.0,
                "speaker": "SPEAKER_02",
                "text": "goodbye",
            },
        ]
        score, diffs = meeting_eval.turn_text_accuracy(ref, hyp)
        assert score == 1.0
        assert len(diffs) == 2
        assert all(row["ratio"] == 1.0 for row in diffs)


class TestVbxMeetingParams:
    def test_accuracy_params_meeting_branch_raises_fa(self, monkeypatch):
        monkeypatch.setattr(diarization, "MODEL_ID", "pyannote/speaker-diarization-community-1")
        params = diarization._accuracy_mode_params(11, 5380.0)
        clustering = params["clustering"]
        assert clustering["Fa"] == 0.32
        assert clustering["threshold"] == 0.56

    def test_accuracy_params_small_groups_unchanged(self, monkeypatch):
        monkeypatch.setattr(diarization, "MODEL_ID", "pyannote/speaker-diarization-community-1")
        params = diarization._accuracy_mode_params(4, 200.0)
        assert params["clustering"]["Fa"] == 0.22
        assert params["clustering"]["threshold"] == 0.54

    def test_retry_params_raise_fa_on_vbx(self, monkeypatch):
        monkeypatch.setattr(diarization, "MODEL_ID", "pyannote/speaker-diarization-community-1")
        base = {"clustering": {"threshold": 0.60, "Fa": 0.20, "Fb": 0.8}}
        retry = diarization._retry_pipeline_params(base)
        assert retry["clustering"]["Fa"] > 0.20

    def test_curated_meeting_configs_vary_fa(self):
        configs = sampling._curated_accuracy_configs(11)
        fas = {cfg["clustering"]["Fa"] for _, cfg in configs}
        assert len(fas) >= 3


class TestCentroidMerge:
    def _segments(self):
        return [
            {"start": 0.0, "end": 60.0, "speaker": "SPEAKER_00"},
            {"start": 60.0, "end": 90.0, "speaker": "SPEAKER_01"},
            {"start": 90.0, "end": 100.0, "speaker": "SPEAKER_02"},
        ]

    def test_merges_same_voice_clusters(self, monkeypatch):
        import numpy as np

        monkeypatch.setenv("DIARIZATION_CENTROID_MERGE_THRESHOLD", "0.60")
        # 00 and 02 are the same voice; 01 is orthogonal.
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.98, 0.20, 0.0],
        ])
        diarization._last_pass_embeddings.clear()
        diarization._last_pass_embeddings.append(
            (["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"], embeddings)
        )
        merged = diarization._merge_similar_speaker_clusters(self._segments())
        speakers = {seg["speaker"] for seg in merged}
        assert speakers == {"SPEAKER_00", "SPEAKER_01"}
        assert merged[2]["speaker"] == "SPEAKER_00"  # absorbed into longer cluster
        assert not diarization._last_pass_embeddings  # stash consumed

    def test_label_mismatch_is_noop(self, monkeypatch):
        import numpy as np

        monkeypatch.setenv("DIARIZATION_CENTROID_MERGE_THRESHOLD", "0.60")
        diarization._last_pass_embeddings.clear()
        diarization._last_pass_embeddings.append(
            (["SPEAKER_07", "SPEAKER_08"], np.eye(2))
        )
        segments = self._segments()
        assert diarization._merge_similar_speaker_clusters(segments) == segments

    def test_disabled_by_default_outside_accuracy_mode(self, monkeypatch):
        import numpy as np

        monkeypatch.delenv("DIARIZATION_CENTROID_MERGE_THRESHOLD", raising=False)
        monkeypatch.setenv("DIARIZATION_ACCURACY_MODE", "false")
        diarization._last_pass_embeddings.clear()
        diarization._last_pass_embeddings.append(
            (["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"], np.eye(3))
        )
        segments = self._segments()
        assert diarization._merge_similar_speaker_clusters(segments, 11) == segments

    def test_default_gated_to_meeting_scale(self, monkeypatch):
        monkeypatch.delenv("DIARIZATION_CENTROID_MERGE_THRESHOLD", raising=False)
        monkeypatch.setenv("DIARIZATION_ACCURACY_MODE", "true")
        assert diarization._centroid_merge_threshold(4) == 0.0
        assert diarization._centroid_merge_threshold(11) == 0.72


class TestOverclusterKwargs:
    def test_meeting_scale_requests_extra_clusters(self, monkeypatch):
        monkeypatch.setattr(diarization, "MODEL_ID", "pyannote/speaker-diarization-community-1")
        monkeypatch.setenv("DIARIZATION_ACCURACY_MODE", "true")
        monkeypatch.delenv("DIARIZATION_OVERCLUSTER_EXTRA", raising=False)
        kwargs = diarization._build_diarize_kwargs(0, 11)
        assert kwargs == {"num_speakers": 18}

    def test_small_groups_keep_upper_bound_semantics(self, monkeypatch):
        monkeypatch.setattr(diarization, "MODEL_ID", "pyannote/speaker-diarization-community-1")
        monkeypatch.setenv("DIARIZATION_ACCURACY_MODE", "true")
        kwargs = diarization._build_diarize_kwargs(0, 4)
        assert kwargs == {"num_speakers": 5}

    def test_min_hint_disables_overcluster(self, monkeypatch):
        monkeypatch.setattr(diarization, "MODEL_ID", "pyannote/speaker-diarization-community-1")
        monkeypatch.setenv("DIARIZATION_ACCURACY_MODE", "true")
        kwargs = diarization._build_diarize_kwargs(0, 11, min_speakers_hint=2)
        assert kwargs == {"max_speakers": 11, "min_speakers": 2}

    def test_env_override_disables(self, monkeypatch):
        monkeypatch.setattr(diarization, "MODEL_ID", "pyannote/speaker-diarization-community-1")
        monkeypatch.setenv("DIARIZATION_ACCURACY_MODE", "true")
        monkeypatch.setenv("DIARIZATION_OVERCLUSTER_EXTRA", "0")
        kwargs = diarization._build_diarize_kwargs(0, 11)
        assert kwargs == {"max_speakers": 11}


class TestRebrandSpanReplacement:
    def test_chronological_sub_voice_maps_to_global_cycle(self):
        span = {"start": 100.0, "end": 200.0, "speaker": "G3"}
        replacement = [
            {"start": 100.0, "end": 170.0, "speaker": "SPEAKER_00"},
            {"start": 170.0, "end": 180.0, "speaker": "SPEAKER_01"},
            {"start": 180.0, "end": 200.0, "speaker": "SPEAKER_00"},
        ]
        context = [
            {"start": 0.0, "end": 50.0, "speaker": "G2"},
            {"start": 50.0, "end": 100.0, "speaker": "G3"},
            {"start": 200.0, "end": 300.0, "speaker": "G4"},
        ]
        out = diarization._rebrand_span_replacement(
            replacement, span, 2, context_segments=context,
        )
        assert out[0]["speaker"] == "G3"
        assert out[2]["speaker"] == "G3"
        assert out[1]["speaker"] == "G4"

    def test_empty_replacement_passthrough(self):
        span = {"start": 0.0, "end": 10.0, "speaker": "G0"}
        assert diarization._rebrand_span_replacement([], span, 0) == []
