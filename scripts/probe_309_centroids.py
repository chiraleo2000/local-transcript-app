#!/usr/bin/env python3
"""One pyannote pass on 309.m4a: dump centroids, sweep centroid-merge offline.

Prints the inter-cluster cosine similarity matrix, the reference mapping, and
the meeting-eval scores that each DIARIZATION_CENTROID_MERGE_THRESHOLD value
would produce — without re-running the GPU pass per threshold.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
AUDIO = REPO / "tests" / "309.m4a"
EXPECTED = REPO / "tests" / "309.txt"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fa", type=float, default=0.20)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--max-speakers", type=int, default=13)
    parser.add_argument("--num-speakers", type=int, default=0,
                        help="exact cluster count (forces KMeans partition)")
    parser.add_argument("--seconds", type=float, default=0.0)
    return parser.parse_args()


def _load_reference_turns(seconds: float):
    from tests.golden.meeting_eval import load_reference_turns

    ref = load_reference_turns(EXPECTED, total_duration_s=5380.6)
    if seconds <= 0:
        return ref
    return [
        {
            "start": turn["start"],
            "end": min(turn["end"], seconds),
            "speaker": turn["speaker"],
        }
        for turn in ref
        if turn["start"] < seconds
    ]


def _build_pipeline(args: argparse.Namespace):
    import torch

    from engines.diarization import load_offline_pyannote_pipeline

    pipe = load_offline_pyannote_pipeline()
    pipe.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    pipe.instantiate({
        "segmentation": {"min_duration_off": 0.04},
        "clustering": {"threshold": args.threshold, "Fa": args.fa, "Fb": 0.8},
    })
    return pipe


def _prepare_audio_source(seconds: float) -> str:
    if seconds <= 0:
        return str(AUDIO)
    clip = Path(tempfile.gettempdir()) / f"probe309_{int(seconds)}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-t", str(seconds), "-i", str(AUDIO),
         "-ac", "1", "-ar", "16000", str(clip)],
        check=True,
    )
    return str(clip)


def _run_diarization(pipe, source: str, args: argparse.Namespace):
    from engines.diarization import _prepare_audio_for_pyannote

    audio_input = _prepare_audio_for_pyannote(source)
    if args.num_speakers > 0:
        return pipe(audio_input, num_speakers=args.num_speakers)
    return pipe(audio_input, max_speakers=args.max_speakers)


def _save_centroids(output, args: argparse.Namespace) -> tuple[list[str], object, list[dict]]:
    import numpy as np

    annotation = output.exclusive_speaker_diarization
    labels = [str(label) for label in annotation.labels()]
    centroids = np.asarray(output.speaker_embeddings, dtype=float)
    segments = [
        {"start": float(turn.start), "end": float(turn.end), "speaker": str(spk)}
        for turn, _, spk in annotation.itertracks(yield_label=True)
    ]
    tag = f"fa{int(args.fa * 100)}" + (f"_n{args.num_speakers}" if args.num_speakers else "")
    out = REPO / "tests" / "output" / f"309_centroids_{tag}.json"
    out.write_text(json.dumps({
        "labels": labels,
        "centroids": centroids.tolist(),
        "segments": segments,
    }, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {out}")
    return labels, centroids, segments


def _print_similarity_matrix(labels: list[str], centroids):
    import numpy as np

    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = centroids / norms
    sim = unit @ unit.T
    print("=== cosine similarity (upper triangle) ===")
    header = "        " + " ".join(f"{lb[-2:]:>5s}" for lb in labels)
    print(header)
    for i, lb in enumerate(labels):
        cells = " ".join(
            f"{sim[i, j]:5.2f}" if j > i else "     " for j in range(len(labels))
        )
        print(f"{lb:>8s} {cells}")
    return sim


def _merged_segments_at_threshold(
    labels: list[str],
    sim,
    segments: list[dict],
    threshold: float,
) -> list[dict]:
    parent = {lb: lb for lb in labels}

    def find(lb: str) -> str:
        while parent[lb] != lb:
            parent[lb] = parent[parent[lb]]
            lb = parent[lb]
        return lb

    durations: dict[str, float] = {}
    for seg in segments:
        durations[seg["speaker"]] = durations.get(seg["speaker"], 0.0) + (
            seg["end"] - seg["start"]
        )
    order = sorted(range(len(labels)), key=lambda i: -durations.get(labels[i], 0.0))
    for pos, i in enumerate(order):
        for j in order[pos + 1:]:
            if sim[i, j] >= threshold and find(labels[j]) != find(labels[i]):
                parent[find(labels[j])] = find(labels[i])
    return [{**seg, "speaker": find(seg["speaker"])} for seg in segments]


def _sweep_merge_thresholds(
    labels: list[str],
    sim,
    segments: list[dict],
    ref_turns: list[dict],
) -> None:
    from tests.golden.meeting_eval import evaluate_meeting_diarization

    print("=== merge threshold sweep (offline) ===")
    for threshold in (1.01, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45):
        merged = _merged_segments_at_threshold(labels, sim, segments, threshold)
        rep = evaluate_meeting_diarization(ref_turns, merged)
        print(json.dumps({
            "merge_t": threshold if threshold <= 1 else "off",
            "detected": rep["detected_speakers"],
            "expected": rep["expected_speakers"],
            "time_acc": rep["speaker_time_accuracy"],
            "turn_acc": rep["turn_accuracy"],
            "b2s": rep["boundary_within_2s"],
        }))


def main() -> int:
    args = _parse_args()
    from _bootstrap import bootstrap

    bootstrap()

    ref = _load_reference_turns(args.seconds)
    pipe = _build_pipeline(args)
    source = _prepare_audio_source(args.seconds)
    output = _run_diarization(pipe, source, args)
    labels, centroids, segments = _save_centroids(output, args)
    sim = _print_similarity_matrix(labels, centroids)
    _sweep_merge_thresholds(labels, sim, segments, ref)
    return 0


if __name__ == "__main__":
    from _gpu_queue import run_locked

    raise SystemExit(run_locked(main))
