# Vendored: Respiro-en

Upstream: <https://github.com/ydqmkkx/Respiro-en>, commit `70e01c60c2f582c41092730680f2894ab24d6467`
Paper: Dong Yang, Tomoki Koriyama, Yuki Saito, "Frame-Wise Breath Detection with Self-Training: An Exploration of
Enhancing Breath Naturalness in Text-to-Speech", Interspeech 2024 (arXiv 2402.00288).
Licence: MIT (`LICENSE`, unmodified).

Every file here was read in full and is **unmodified** from that commit (git blob hashes match upstream's tree):

| file | SHA-256 |
|---|---|
| `modules.py` | `f789e0986e3090d7df5f9f0f596d9e3601c6da514c3ac01a65920a493b840e46` |
| `LICENSE` | `a34ad1af58dc7c02f867f620f7ddc952029b383c9b0dce349d54f6b875e079cd` |

`modules.py` is the reference: fpab runs the model with its own numpy port (`breath_np.py`). The tests hold the
network to this code within 1e-5 (a small random network built from these classes) and the features to librosa within
0.002 dB; with the real weights, whole chapters matched within 1e-5, no frame changing side of 0.5.

## The weights

Not vendored. `fpab setup-model` installs `respiro-en-fpab-v2.safetensors` (11,713,964 bytes, SHA-256
`17dc2cb7bac027a2d5aa72649e575936730790785fe9d2edef6a82206dcf587d`) from this repository's `breath-model-v2`
release and refuses any file that does not match.

They are the published `respiro-en.pt` (SHA-256
`1f4a9b96f96645c480bf0e07b1e18cd68878ac0b4bb5dc920ad93f9b17df858a`; its pickle references only OrderedDict and
torch's tensor-rebuild functions, and was loaded once with `torch.load(weights_only=True)`) fine-tuned with the
paper's recipe (AdamW, peak learning rate 2e-5, warm-up then linear decay, frame-wise binary cross-entropy) on
narration breaths: one title's breaths labelled by ear, plus the published model's confident verdicts on part of a
second title. Only the network's float32 tensors are published — no metadata, no audio, no labels, no names; a
per-frame breath classifier cannot give back the audio it was trained on. The epoch was chosen on held-back data
by a rule fixed before training, and the model was then judged on tagged events never used for either.
