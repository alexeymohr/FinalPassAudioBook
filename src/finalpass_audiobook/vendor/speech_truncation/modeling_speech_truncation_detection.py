from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import LlamaConfig, LlamaModel, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

try:
    from .configuration_speech_truncation_detection import SpeechTruncationDetectionConfig
    from .processing_speech_truncation_detection import SpeechTruncationDetectionProcessor
except ImportError:
    from configuration_speech_truncation_detection import SpeechTruncationDetectionConfig
    from processing_speech_truncation_detection import SpeechTruncationDetectionProcessor


@dataclass
class SpeechTruncationPrediction:
    logits: torch.Tensor
    truncation_probabilities: torch.Tensor
    truncation_score: torch.Tensor
    confidence_score: torch.Tensor
    is_truncated: torch.Tensor
    threshold: float
    decision_window_frames: int


class _ConfigurableCausalityLlamaModel(LlamaModel):
    """LlamaModel wrapper with optional bidirectional (non-causal) attention."""

    def __init__(self, config: LlamaConfig, *, disable_causal_attention: bool):
        super().__init__(config)
        self.disable_causal_attention = bool(disable_causal_attention)
        self._apply_attention_causality()

    def _apply_attention_causality(self) -> None:
        use_causal_attention = not self.disable_causal_attention
        setattr(self.config, "is_causal", bool(use_causal_attention))

        for layer in getattr(self, "layers", []):
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is not None and hasattr(self_attn, "is_causal"):
                setattr(self_attn, "is_causal", bool(use_causal_attention))

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor | None,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Any,
        output_attentions: bool,
    ) -> torch.Tensor | None:
        if not self.disable_causal_attention:
            parent_update = getattr(super(), "_update_causal_mask", None)
            if parent_update is None:
                return attention_mask
            return parent_update(
                attention_mask=attention_mask,
                input_tensor=input_tensor,
                cache_position=cache_position,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
            )

        attn_impl = str(getattr(self.config, "_attn_implementation", ""))
        if attn_impl in {"flash_attention_2", "flash_attention_3", "flex_attention"}:
            if attention_mask is None:
                return None
            if attention_mask.ndim == 2:
                return attention_mask.to(device=input_tensor.device, dtype=torch.bool)
            if attention_mask.ndim == 4:
                return attention_mask.to(device=input_tensor.device)
            raise ValueError(
                "Expected attention_mask shape [batch, time] or [batch, 1, query, key] "
                f"for non-causal mode, got {tuple(attention_mask.shape)}"
            )

        batch_size, query_length = int(input_tensor.shape[0]), int(input_tensor.shape[1])
        dtype = input_tensor.dtype
        device = input_tensor.device

        if attention_mask is None:
            return torch.zeros((batch_size, 1, query_length, query_length), dtype=dtype, device=device)

        if attention_mask.ndim == 4:
            return attention_mask.to(device=device, dtype=dtype)

        if attention_mask.ndim != 2:
            raise ValueError(
                "Expected attention_mask shape [batch, time] or [batch, 1, query, key] "
                f"for non-causal mode, got {tuple(attention_mask.shape)}"
            )

        key_mask = attention_mask.to(device=device, dtype=dtype)
        key_mask = key_mask[:, None, None, :]
        additive_mask = (1.0 - key_mask) * torch.finfo(dtype).min
        return additive_mask.expand(batch_size, 1, query_length, int(key_mask.shape[-1])).contiguous()


class SpeechTruncationDetectionModel(PreTrainedModel):
    config_class = SpeechTruncationDetectionConfig
    base_model_prefix = ""

    def __init__(self, config: SpeechTruncationDetectionConfig) -> None:
        super().__init__(config)

        model_config = dict(config.model_config)
        audio_config = dict(config.audio_config)

        self.num_labels = int(model_config.get("num_labels", 2))
        self.label_ignore_index = int(config.label_ignore_index)
        self.input_num_mels = int(audio_config["n_mels"])
        self.label_smoothing = float(model_config.get("label_smoothing", 0.02))

        llama_payload: dict[str, Any] = dict(model_config.get("llama_config", {}))
        llama_payload["use_cache"] = False
        llama_config = LlamaConfig(**llama_payload)
        requested_attn_impl = str(model_config.get("attn_implementation", "flash_attention_2"))
        attn_impl = str(requested_attn_impl)
        if (not torch.cuda.is_available()) and attn_impl in {"flash_attention_2", "flex_attention"}:
            attn_impl = "sdpa"
        setattr(llama_config, "_attn_implementation", attn_impl)
        if hasattr(llama_config, "attn_implementation"):
            setattr(llama_config, "attn_implementation", attn_impl)

        disable_causal_attention = bool(model_config.get("disable_causal_attention", False))
        try:
            self.backbone = _ConfigurableCausalityLlamaModel(
                llama_config,
                disable_causal_attention=disable_causal_attention,
            )
        except Exception as exc:
            if attn_impl not in {"flash_attention_2", "flex_attention"}:
                raise
            fallback_llama_config = LlamaConfig(**llama_payload)
            setattr(fallback_llama_config, "_attn_implementation", "sdpa")
            if hasattr(fallback_llama_config, "attn_implementation"):
                setattr(fallback_llama_config, "attn_implementation", "sdpa")
            self.backbone = _ConfigurableCausalityLlamaModel(
                fallback_llama_config,
                disable_causal_attention=disable_causal_attention,
            )
            attn_impl = "sdpa"
            self.config.model_config["attn_fallback_error"] = f"{type(exc).__name__}: {exc}"

        self.active_attn_implementation = str(attn_impl)
        self.config.model_config["attn_implementation_runtime"] = self.active_attn_implementation
        self.input_projection = nn.Linear(
            self.input_num_mels,
            int(llama_config.hidden_size),
            bias=bool(model_config.get("input_projection_bias", True)),
        )
        self.classifier_dropout = nn.Dropout(float(model_config.get("classifier_dropout", 0.1)))
        self.classifier = nn.Linear(int(llama_config.hidden_size), self.num_labels)

        self.post_init()

    def forward(
        self,
        *,
        mel: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> TokenClassifierOutput | tuple[torch.Tensor | None, torch.Tensor]:
        if mel.ndim != 3:
            raise ValueError(f"Expected mel shape [batch, time, n_mels], got shape={tuple(mel.shape)}")

        inputs_embeds = self.input_projection(mel)

        attn_mask = None
        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError(
                    "Expected attention_mask shape [batch, time], "
                    f"got shape={tuple(attention_mask.shape)}"
                )
            attn_mask = attention_mask.to(dtype=torch.bool)

        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            use_cache=False,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.classifier(self.classifier_dropout(hidden_states))

        loss = None
        if labels is not None:
            if labels.shape != logits.shape[:2]:
                raise ValueError(
                    "labels shape must match logits[:2] "
                    f"(labels={tuple(labels.shape)}, logits={tuple(logits.shape)})"
                )
            loss = F.cross_entropy(
                logits.reshape(-1, self.num_labels),
                labels.reshape(-1),
                ignore_index=self.label_ignore_index,
                label_smoothing=float(self.label_smoothing),
            )

        if not return_dict:
            return loss, logits
        return TokenClassifierOutput(loss=loss, logits=logits)

    def get_processor(self, *, tail_seconds: float | None = None) -> SpeechTruncationDetectionProcessor:
        return SpeechTruncationDetectionProcessor.from_config(self.config, tail_seconds=tail_seconds)

    @torch.inference_mode()
    def predict_probabilities(
        self,
        *,
        mel: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.forward(mel=mel, attention_mask=attention_mask, labels=None, return_dict=True)
        return torch.softmax(outputs.logits, dim=-1)

    @torch.inference_mode()
    def predict_truncation_mask(
        self,
        *,
        mel: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        probs = self.predict_probabilities(mel=mel, attention_mask=attention_mask)
        truncated_prob = probs[..., 1]
        return (truncated_prob >= float(threshold)).to(dtype=torch.long)

    def _resolve_decision_window_frames(self, *, decision_window_ms: float | None) -> int:
        audio_config = dict(self.config.audio_config)
        target_sr = int(audio_config["target_sample_rate"])
        hop = int(audio_config["hop_length"])
        frame_stride_s = float(hop) / float(target_sr)
        resolved_window_ms = (
            float(self.config.inference.get("decision_window_ms", 30.0))
            if decision_window_ms is None
            else float(decision_window_ms)
        )
        if resolved_window_ms <= 0.0:
            raise ValueError("decision_window_ms must be > 0")
        return max(1, int(math.ceil((resolved_window_ms / 1000.0) / frame_stride_s)))

    def score_truncation_probabilities(
        self,
        *,
        truncation_probabilities: torch.Tensor,
        decision_window_ms: float | None = None,
        threshold: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if truncation_probabilities.ndim != 2:
            raise ValueError(
                "Expected truncation_probabilities shape [batch, time], "
                f"got {tuple(truncation_probabilities.shape)}"
            )
        decision_window_frames = self._resolve_decision_window_frames(decision_window_ms=decision_window_ms)
        resolved_threshold = (
            float(self.config.inference.get("threshold", 0.5))
            if threshold is None
            else float(threshold)
        )
        if not (0.0 <= resolved_threshold <= 1.0):
            raise ValueError("threshold must be in [0, 1]")

        tail_window = truncation_probabilities[:, -int(decision_window_frames) :]
        scores = torch.max(tail_window, dim=1).values
        decisions = scores >= float(resolved_threshold)
        return scores, decisions

    @torch.inference_mode()
    def predict_from_audio(
        self,
        *,
        audio: Any,
        sampling_rate: int | list[int] | torch.Tensor | None = None,
        tail_seconds: float | None = None,
        decision_window_ms: float | None = None,
        threshold: float | None = None,
        processor: SpeechTruncationDetectionProcessor | None = None,
    ) -> SpeechTruncationPrediction:
        model_device = next(self.parameters()).device
        active_processor = processor if processor is not None else self.get_processor(tail_seconds=tail_seconds)
        batch = active_processor(audio=audio, sampling_rate=sampling_rate, device=model_device)

        output = self.forward(
            mel=batch["mel"],
            attention_mask=batch["attention_mask"],
            labels=None,
            return_dict=True,
        )
        logits = output.logits
        truncation_probabilities = torch.softmax(logits, dim=-1)[..., 1]

        scores, decisions = self.score_truncation_probabilities(
            truncation_probabilities=truncation_probabilities,
            decision_window_ms=decision_window_ms,
            threshold=threshold,
        )
        resolved_threshold = (
            float(self.config.inference.get("threshold", 0.5))
            if threshold is None
            else float(threshold)
        )
        decision_window_frames = self._resolve_decision_window_frames(decision_window_ms=decision_window_ms)

        return SpeechTruncationPrediction(
            logits=logits,
            truncation_probabilities=truncation_probabilities,
            truncation_score=scores,
            confidence_score=scores,
            is_truncated=decisions,
            threshold=float(resolved_threshold),
            decision_window_frames=int(decision_window_frames),
        )
