"""Tests for adaptive ASR performance policy."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from backend.asr_performance import (
    adaptive_num_beams,
    adaptive_turn_merge_gap_s,
    apply_performance_policy,
    performance_target_seconds,
)


class TestPerformanceTarget(unittest.TestCase):
    def test_short_audio_ten_minute_cap(self) -> None:
        self.assertAlmostEqual(performance_target_seconds(210.0), 600.0)

    def test_under_fifteen_minutes_stays_at_ten_minute_cap(self) -> None:
        self.assertAlmostEqual(performance_target_seconds(14 * 60), 600.0)

    def test_fifteen_minute_audio_two_thirds_realtime(self) -> None:
        # 15 min / 1.5 = 10 min wall
        self.assertAlmostEqual(performance_target_seconds(15 * 60), 10 * 60)

    def test_ninety_minute_audio_two_thirds_realtime(self) -> None:
        # 90 min / 1.5 = 60 min wall
        self.assertAlmostEqual(performance_target_seconds(90 * 60), 60 * 60)


class TestAdaptiveBeams(unittest.TestCase):
    def test_short_clip_uses_max_beams(self) -> None:
        with patch.dict(
            os.environ,
            {"ASR_NUM_BEAMS_MAX": "8", "ASR_NUM_BEAMS_MIN": "4"},
            clear=True,
        ):
            self.assertEqual(adaptive_num_beams(3.5 * 60, diarization=True), 8)

    def test_long_clip_scales_down(self) -> None:
        with patch.dict(
            os.environ,
            {"ASR_NUM_BEAMS_MAX": "8", "ASR_NUM_BEAMS_MIN": "4"},
            clear=True,
        ):
            self.assertLessEqual(adaptive_num_beams(25 * 60, diarization=True), 5)


class TestApplyPolicy(unittest.TestCase):
    def test_apply_sets_env(self) -> None:
        env = {
            "ASR_ADAPTIVE_PERFORMANCE": "true",
            "ASR_NUM_BEAMS_MAX": "8",
            "ASR_NUM_BEAMS_MIN": "4",
            "ASR_QUALITY_PROFILE": "high",
            "DIARIZATION_ACCURACY_MODE": "true",
            "ASR_TURN_GUIDED_MERGE_GAP_S": "0.25",
        }
        with patch.dict(os.environ, env, clear=True):
            applied = apply_performance_policy(210.0, diarization=True)
            self.assertIn("ASR_NUM_BEAMS", applied)
            self.assertEqual(applied["ASR_NUM_BEAMS"], "8")
            self.assertEqual(adaptive_turn_merge_gap_s(210.0), 0.25)

    def test_long_audio_policy(self) -> None:
        env = {
            "ASR_ADAPTIVE_PERFORMANCE": "true",
            "ASR_TARGET_LONG_AUDIO_S": "3600",
            "ASR_DIAR_WINDOWED_FAST": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            applied = apply_performance_policy(77 * 60, diarization=True)
        self.assertEqual(applied.get("ASR_TURN_GUIDED"), "true")
        self.assertEqual(applied.get("ASR_NUM_BEAMS"), "4")
        self.assertNotIn("ASR_LONG_FORM_WINDOW_S", applied)

    def test_disabled_when_flag_off(self) -> None:
        with patch.dict(os.environ, {"ASR_ADAPTIVE_PERFORMANCE": "false"}, clear=False):
            applied = apply_performance_policy(210.0, diarization=True)
        self.assertEqual(applied, {})


if __name__ == "__main__":
    unittest.main()
