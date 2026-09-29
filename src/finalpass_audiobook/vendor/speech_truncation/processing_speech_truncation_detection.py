from __future__ import annotations

import numbers
from typing import Any, Iterable

import numpy as np
import torch
import torchaudio

try:
    from .configuration_speech_truncation_detection import SpeechTruncationDetectionConfig
except ImportError:
    from configuration_speech_truncation_detection import SpeechTruncationDetectionConfig

STYLETTS_MEL_LOG_EPS = 1e-5
STYLETTS_MEL_MEAN = -4.0
STYLETTS_MEL_STD = 4.0


class SpeechTruncationDetectionProcessor:
    """Inference-time preprocessing for truncation detection.

    Converts flexible waveform inputs into model-ready batches:
    - decode/normalize inputs to mono float tensors
    - optional resample to configured target_sample_rate
    - fixed tail crop with left pad to `tail_seconds`
    - mel/log normalization matching training + pipeline inference
    """

    def __init__(
        self,
        config: SpeechTruncationDetectionConfig,
        *,
        tail_seconds: float | None = None,
    ) -> None:
        self.config = config
        self.audio_config = dict(config.audio_config)
        self.inference_config = dict(config.inference)

        self.target_sample_rate = int(self.audio_config["target_sample_rate"])
        self.n_fft = int(self.audio_config["n_fft"])
        self.win_length = int(self.audio_config["win_length"])
        self.hop_length = int(self.audio_config["hop_length"])
        self.n_mels = int(self.audio_config["n_mels"])
        self.f_min = float(self.audio_config.get("f_min", 0.0))
        self.f_max = self.audio_config.get("f_max")
        self.mel_power = float(self.audio_config.get("mel_power", 2.0))
        self.window_fn_name = str(self.audio_config.get("window_fn", "hann"))
        self.center = bool(self.audio_config.get("center", True))
        self.pad_mode = str(self.audio_config.get("pad_mode", "constant"))
        self.mel_log_eps = float(self.audio_config.get("mel_log_eps", 1e-6))
        self.use_styletts_mel_normalization = bool(self.audio_config.get("use_styletts_mel_normalization", True))

        resolved_tail_seconds = (
            float(self.inference_config.get("tail_seconds", 5.0))
            if tail_seconds is None
            else float(tail_seconds)
        )
        if resolved_tail_seconds <= 0.0:
            raise ValueError("tail_seconds must be > 0")
        self.tail_seconds = float(resolved_tail_seconds)
        sampled_target_num_samples = max(1, int(round(self.tail_seconds * self.target_sample_rate)))
        self.target_num_samples = self._snap_target_num_samples(sampled_target_num_samples)

        self._mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            f_min=self.f_min,
            f_max=self.f_max,
            n_mels=self.n_mels,
            power=self.mel_power,
            window_fn=self._resolve_window_fn(self.window_fn_name),
            center=self.center,
            pad_mode=self.pad_mode,
            norm=None,
        )

    @classmethod
    def from_config(
        cls,
        config: SpeechTruncationDetectionConfig,
        *,
        tail_seconds: float | None = None,
    ) -> "SpeechTruncationDetectionProcessor":
        return cls(config=config, tail_seconds=tail_seconds)

    @staticmethod
    def _resolve_window_fn(name: str):
        if name == "hann":
            return torch.hann_window
        if name == "hamming":
            return torch.hamming_window
        if name == "rectangular":
            return torch.ones
        raise ValueError(f"Unsupported window_fn={name!r}")

    def _snap_target_num_samples(self, sampled_target_num_samples: int) -> int:
        target = max(1, int(sampled_target_num_samples))
        hop = int(self.hop_length)
        if target <= hop:
            return int(hop)
        k = max(1, int(round(float(target) / float(hop))))
        return int(k * hop)

    @staticmethod
    def _to_mono_1d_tensor(value: torch.Tensor) -> torch.Tensor:
        wav = value.detach().to(device="cpu", dtype=torch.float32)
        if wav.ndim == 1:
            return wav.contiguous()
        if wav.ndim == 2:
            if int(wav.shape[0]) == 1:
                return wav[0].contiguous()
            if int(wav.shape[1]) == 1:
                return wav[:, 0].contiguous()
            # Heuristic channel handling for common [channels, time] and [time, channels] layouts.
            if int(wav.shape[0]) <= 8 and int(wav.shape[1]) > int(wav.shape[0]):
                return wav.mean(dim=0).contiguous()
            if int(wav.shape[1]) <= 8 and int(wav.shape[0]) > int(wav.shape[1]):
                return wav.mean(dim=1).contiguous()
        raise ValueError(f"Expected 1D mono or 2D mono/stereo tensor, got shape={tuple(wav.shape)}")

    @classmethod
    def _as_audio_tensor(cls, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return cls._to_mono_1d_tensor(value)
        if isinstance(value, np.ndarray):
            return cls._to_mono_1d_tensor(torch.from_numpy(np.asarray(value, dtype=np.float32)))
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                raise ValueError("audio list item is empty")
            if all(np.isscalar(x) for x in value):
                return cls._to_mono_1d_tensor(torch.as_tensor(value, dtype=torch.float32))
        raise TypeError(
            "Unsupported audio item type="
            f"{type(value).__name__}; expected torch.Tensor, numpy.ndarray, or scalar list/tuple"
        )

    @staticmethod
    def _resolve_sample_rate_list(
        *,
        batch_size: int,
        sampling_rate: int | Iterable[int] | None,
        default_sample_rate: int,
    ) -> list[int]:
        if sampling_rate is None:
            return [int(default_sample_rate)] * int(batch_size)

        if isinstance(sampling_rate, numbers.Integral):
            return [int(sampling_rate)] * int(batch_size)

        if isinstance(sampling_rate, torch.Tensor):
            if sampling_rate.ndim == 0:
                return [int(sampling_rate.item())] * int(batch_size)
            sr_values = [int(x) for x in sampling_rate.detach().cpu().view(-1).tolist()]
        else:
            sr_values = [int(x) for x in sampling_rate]

        if len(sr_values) != int(batch_size):
            raise ValueError(
                f"sampling_rate length mismatch: expected {batch_size}, got {len(sr_values)}"
            )
        return sr_values

    def _normalize_inputs(
        self,
        *,
        audio: Any,
        sampling_rate: int | Iterable[int] | None,
    ) -> list[tuple[torch.Tensor, int]]:
        if isinstance(audio, torch.Tensor):
            if audio.ndim == 1:
                batch = [self._as_audio_tensor(audio)]
            elif audio.ndim == 2:
                batch = [self._as_audio_tensor(audio[idx]) for idx in range(int(audio.shape[0]))]
            else:
                raise ValueError(f"Unsupported torch audio shape={tuple(audio.shape)}")
        elif isinstance(audio, np.ndarray):
            if audio.ndim == 1:
                batch = [self._as_audio_tensor(audio)]
            elif audio.ndim == 2:
                batch = [self._as_audio_tensor(audio[idx]) for idx in range(int(audio.shape[0]))]
            else:
                raise ValueError(f"Unsupported numpy audio shape={tuple(audio.shape)}")
        elif isinstance(audio, (list, tuple)):
            if len(audio) == 0:
                raise ValueError("audio list is empty")
            # Support either [sample0, sample1, ...] as one waveform or a list/tuple batch.
            if all(np.isscalar(x) for x in audio):
                batch = [self._as_audio_tensor(audio)]
            else:
                batch = [self._as_audio_tensor(item) for item in audio]
        else:
            raise TypeError(
                "Unsupported audio container type. Expected tensor/ndarray/list/tuple, "
                f"got {type(audio).__name__}"
            )

        sample_rates = self._resolve_sample_rate_list(
            batch_size=len(batch),
            sampling_rate=sampling_rate,
            default_sample_rate=self.target_sample_rate,
        )
        out: list[tuple[torch.Tensor, int]] = []
        for wav, sr in zip(batch, sample_rates):
            if int(sr) <= 0:
                raise ValueError(f"Invalid sampling rate: {sr}")
            out.append((wav, int(sr)))
        return out

    @staticmethod
    def _tail_crop_with_left_pad(wav: torch.Tensor, *, target_num_samples: int) -> torch.Tensor:
        source_num_samples = int(wav.shape[-1])
        if source_num_samples >= int(target_num_samples):
            return wav[source_num_samples - int(target_num_samples) :]
        left_pad = int(target_num_samples) - source_num_samples
        return torch.nn.functional.pad(wav, (left_pad, 0), mode="constant", value=0.0)

    def _preprocess_single(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        x = wav
        if int(sample_rate) != int(self.target_sample_rate):
            x = torchaudio.functional.resample(
                x,
                orig_freq=int(sample_rate),
                new_freq=int(self.target_sample_rate),
            )
        x = self._tail_crop_with_left_pad(x, target_num_samples=self.target_num_samples)
        return x

    def _maybe_trim_last_center_frame(self, mel: torch.Tensor) -> torch.Tensor:
        if not bool(self.center):
            return mel
        hop = int(self.hop_length)
        wav_len = int(self.target_num_samples)
        if hop <= 0 or (wav_len % hop) != 0:
            return mel

        expected_num_frames = int(wav_len // hop)
        actual_num_frames = int(mel.shape[1])
        if actual_num_frames == (expected_num_frames + 1):
            return mel[:, :-1, :]
        if actual_num_frames != expected_num_frames:
            raise RuntimeError(
                "Unexpected center=True mel frame count in preprocessing; "
                f"expected={expected_num_frames} actual={actual_num_frames} "
                f"wav_num_samples={wav_len} hop={hop}"
            )
        return mel

    def __call__(
        self,
        *,
        audio: Any,
        sampling_rate: int | Iterable[int] | None = None,
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        normalized = self._normalize_inputs(audio=audio, sampling_rate=sampling_rate)
        clipped = [self._preprocess_single(wav, sr) for wav, sr in normalized]
        wav_batch = torch.stack(clipped, dim=0)

        mel = self._mel_transform(wav_batch)
        if self.use_styletts_mel_normalization:
            mel = (torch.log(torch.clamp(mel, min=STYLETTS_MEL_LOG_EPS)) - STYLETTS_MEL_MEAN) / STYLETTS_MEL_STD
        else:
            mel = torch.log(torch.clamp(mel, min=float(self.mel_log_eps)))

        mel = mel.transpose(1, 2).contiguous()
        mel = self._maybe_trim_last_center_frame(mel)

        batch_size = int(mel.shape[0])
        num_frames = int(mel.shape[1])
        attention_mask = torch.ones((batch_size, num_frames), dtype=torch.bool)
        lengths = torch.full((batch_size,), fill_value=num_frames, dtype=torch.long)

        if device is not None:
            mel = mel.to(device=device)
            attention_mask = attention_mask.to(device=device)
            lengths = lengths.to(device=device)

        return {
            "mel": mel,
            "attention_mask": attention_mask,
            "lengths": lengths,
        }
