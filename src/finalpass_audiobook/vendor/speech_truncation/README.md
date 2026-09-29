---
library_name: transformers
pipeline_tag: audio-classification
tags:
  - audio
  - speech
  - truncation-detection
  - custom-code
license: apache-2.0
---

# Speech Truncation Detection (12M)

Frame-wise speech truncation detector for identifying whether audio ends in a likely cutoff.

The model evaluates the tail of each input waveform and produces per-frame truncation probabilities. A final clip-level score is computed from the final portion of the sequence, then thresholded to return a boolean `is_truncated` prediction.

## Intended Use

This model is intended for speech recordings where you want to decide whether the spoken audio appears to end abruptly (truncated) versus ending naturally.

Use this for speech-oriented pipelines such as dataset quality checks, ingestion validation, or gating before downstream ASR/transcript correction steps.

This model is not intended for non-speech audio classification tasks (music, environmental sounds, general acoustic event detection).

## Quick Start

```python
import numpy as np
from transformers import AutoModel

repo_id = "mythicinfinity/speech-truncation-detection-12M"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
model.eval()

sr = 16000
wav = np.zeros(sr * 2, dtype=np.float32)
out = model.predict_from_audio(audio=wav, sampling_rate=sr)
is_truncated = bool(out.is_truncated[0].item())
confidence = float(out.confidence_score[0].item())
print(is_truncated, confidence)
```

## Usage Examples

### Single waveform

```python
import numpy as np
from transformers import AutoModel

model = AutoModel.from_pretrained("mythicinfinity/speech-truncation-detection-12M", trust_remote_code=True)
waveform = np.random.randn(32000).astype('float32')  # ~2s at 16kHz
out = model.predict_from_audio(audio=waveform, sampling_rate=16000)
print(float(out.confidence_score[0]), bool(out.is_truncated[0]))
```

### Batched tensor of waveforms (`[batch, time]`)

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("mythicinfinity/speech-truncation-detection-12M", trust_remote_code=True)
audio_batch = torch.randn(4, 32000)  # 4 clips, each ~2s at 16kHz
out = model.predict_from_audio(audio=audio_batch, sampling_rate=16000)
print(out.confidence_score.shape)  # torch.Size([4])
print(out.is_truncated.shape)      # torch.Size([4])
```

### List of waveforms (variable lengths) with per-item sample rates

```python
import numpy as np
from transformers import AutoModel

model = AutoModel.from_pretrained("mythicinfinity/speech-truncation-detection-12M", trust_remote_code=True)
audio_list = [
    np.random.randn(24000).astype('float32'),  # 1.5s @ 16kHz
    np.random.randn(8000).astype('float32'),   # 1.0s @ 8kHz
]
sample_rates = [16000, 8000]
out = model.predict_from_audio(audio=audio_list, sampling_rate=sample_rates)
print(out.confidence_score)
print(out.is_truncated)
```

Accepted input forms: single tensor/ndarray waveform, batched tensor (`[B, T]`), or list/tuple of waveforms.

## Outputs

- `truncation_probabilities`: per-frame probability of truncation (`[batch, time]`)
- `confidence_score`: clip-level truncation score (`[batch]`)
- `is_truncated`: clip-level boolean decision (`[batch]`)

## Non-Remote-Code Fallback

If you prefer not to use `trust_remote_code=True`, download the artifact files (`config.json`, `model.safetensors`, `configuration_*.py`, `modeling_*.py`, `processing_*.py`) into a local directory and load from those local files:

```python
import json
import sys
from pathlib import Path
from safetensors.torch import load_file

artifact_dir = Path('/path/to/downloaded-artifact')
sys.path.insert(0, str(artifact_dir))

from configuration_speech_truncation_detection import SpeechTruncationDetectionConfig
from modeling_speech_truncation_detection import SpeechTruncationDetectionModel

cfg_payload = json.loads((artifact_dir / 'config.json').read_text(encoding='utf-8'))
cfg = SpeechTruncationDetectionConfig(**cfg_payload)
model = SpeechTruncationDetectionModel(cfg)
state = load_file(str(artifact_dir / 'model.safetensors'), device='cpu')
model.load_state_dict(state, strict=True)
model.eval()
```
