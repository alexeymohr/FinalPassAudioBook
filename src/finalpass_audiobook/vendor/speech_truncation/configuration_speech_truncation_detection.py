from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class SpeechTruncationDetectionConfig(PretrainedConfig):
    model_type = "speech_truncation_detection"

    def __init__(
        self,
        model_config: dict[str, Any] | None = None,
        audio_config: dict[str, Any] | None = None,
        label_ignore_index: int = -100,
        inference: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        self.model_config = dict(model_config or {})
        self.audio_config = dict(audio_config or {})
        self.label_ignore_index = int(label_ignore_index)
        self.inference = self._build_inference_config(inference)
        super().__init__(**kwargs)

    @staticmethod
    def _build_inference_config(inference: dict[str, Any] | None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "tail_seconds": 5.0,
            "decision_window_ms": 30.0,
            "threshold": 0.5,
            "policy": "tail_max_prob_threshold",
        }
        if inference:
            out.update(dict(inference))

        out["tail_seconds"] = float(out["tail_seconds"])
        out["decision_window_ms"] = float(out["decision_window_ms"])
        out["threshold"] = float(out["threshold"])
        out["policy"] = str(out["policy"])

        if out["tail_seconds"] <= 0.0:
            raise ValueError("inference.tail_seconds must be > 0")
        if out["decision_window_ms"] <= 0.0:
            raise ValueError("inference.decision_window_ms must be > 0")
        if not (0.0 <= out["threshold"] <= 1.0):
            raise ValueError("inference.threshold must be in [0, 1]")
        if out["policy"] != "tail_max_prob_threshold":
            raise ValueError("Only policy='tail_max_prob_threshold' is supported")
        return out
