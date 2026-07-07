"""Compatibility shims for optimum-intel OpenVINO Whisper + transformers 4.57+."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_applied = False


def apply_openvino_whisper_compat() -> None:
    """Map transformers `input_ids` to optimum-intel `decoder_input_ids` for Whisper OV."""
    global _applied
    if _applied:
        return
    try:
        from optimum.intel.openvino.modeling_seq2seq import _OVModelForWhisper
    except ImportError:
        return

    original = _OVModelForWhisper.prepare_inputs_for_generation

    def _compat_prepare_inputs_for_generation(
        self,
        decoder_input_ids=None,
        input_ids=None,
        **kwargs,
    ):
        if decoder_input_ids is None:
            decoder_input_ids = input_ids
        if decoder_input_ids is None:
            decoder_input_ids = kwargs.pop("input_ids", None)
        if decoder_input_ids is None:
            raise ValueError(
                "decoder_input_ids or input_ids must be provided for OpenVINO Whisper generation"
            )
        return original(self, decoder_input_ids, **kwargs)

    _OVModelForWhisper.prepare_inputs_for_generation = _compat_prepare_inputs_for_generation
    _applied = True
    logger.info("Applied OpenVINO Whisper prepare_inputs_for_generation compatibility patch")
