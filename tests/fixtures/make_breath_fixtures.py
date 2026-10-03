"""Write tests/fixtures/breath_np.npz: what the numpy breath port (breath_np) is held to in every test run, with no
PyTorch, torchaudio or librosa there.

* features: librosa's, through the authors' own `feature_extractor` (vendor/respiro/modules.py), for a synthetic
  0.6 s signal (stored, so no random generator has to agree across numpy versions);
* network: a tiny DetectionNet (the authors' own classes and forward, small dimensions, random weights and
  BatchNorm statistics) and its torch output for random features.

Synthetic data only. Needs torch, torchaudio, librosa and intervaltree (not fpab's dependencies); run once, by hand:
    python tests/fixtures/make_breath_fixtures.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchaudio.models import Conformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src" / "finalpass_audiobook" / "vendor" / "respiro"))
from modules import Conv1dUpsampling, Conv2dDownsampling, DetectionNet, feature_extractor  # noqa: E402

TINY = dict(n_mels=16, hidden=8, heads=2, ffn=12, layers=2, kernel=5, lstm=6)

torch.manual_seed(0)
t = np.arange(int(0.6 * 16000)) / 16000
y = (0.3 * np.sin(2 * np.pi * (200 + 300 * t) * t) + 0.05 * np.sin(2 * np.pi * 3100 * t)
     + 0.02 * np.sin(2 * np.pi * 7000 * t) * (np.sin(2 * np.pi * 3 * t) > 0)).astype(np.float32)
y[4000:4400] = 0.0                                       # a stretch of digital silence (zero-crossing rate 0)
feat, _ = feature_extractor(y, 16000)
feat = feat[0].numpy()

m = DetectionNet.__new__(DetectionNet)                   # the authors' forward, with small parts
nn.Module.__init__(m)
reduced = ((TINY["n_mels"] - 3) // 2 + 1 - 3) // 2 + 1
m.downsampling = Conv2dDownsampling(3, 1)
m.upsampling = Conv1dUpsampling(TINY["hidden"], TINY["hidden"])
m.linear = nn.Linear(reduced, TINY["hidden"])
m.dropout = nn.Dropout(0.1)
m.conformer = Conformer(input_dim=TINY["hidden"], num_heads=TINY["heads"], ffn_dim=TINY["ffn"],
                        num_layers=TINY["layers"], depthwise_conv_kernel_size=TINY["kernel"], dropout=0.1)
m.lstm = nn.LSTM(input_size=TINY["hidden"], hidden_size=TINY["lstm"], bidirectional=True, batch_first=True)
m.fc = nn.Linear(2 * TINY["lstm"], 1)
m.sigmoid = nn.Sigmoid()
with torch.no_grad():
    for p in m.parameters():
        p.copy_(torch.randn_like(p) * 0.5)
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm1d):
            mod.running_mean.copy_(torch.randn_like(mod.running_mean) * 0.3)
            mod.running_var.copy_(torch.rand_like(mod.running_var) + 0.5)
m.eval()
x = torch.randn(1, 3, TINY["n_mels"], 101)
with torch.no_grad():
    out = m(x, torch.tensor([101]))[0].numpy()
weights = {k: v.numpy() for k, v in m.state_dict().items() if v.is_floating_point()}
np.savez_compressed(HERE / "breath_np.npz", signal=y, mel_db=feat[0], vms=feat[1, 0], zcr=feat[2, 0],
                    tiny_input=x[0].numpy(), tiny_output=out,
                    **{f"w:{k}": v for k, v in weights.items()}, **{f"cfg:{k}": np.array(v) for k, v in TINY.items()})
print("wrote", HERE / "breath_np.npz", (HERE / "breath_np.npz").stat().st_size, "bytes")
