"""The breath model: install the weights once, load them offline.

The weights are Respiro-en (Yang, Koriyama & Saito, Interspeech 2024; MIT; vendor/respiro) fine-tuned for
narration breaths, published as a release asset of this repository: float32 tensors only, no metadata — nothing in
them names or can give back the audio they were trained on. ``load`` runs them with our numpy port (breath_np; no
PyTorch). Only ``install_from_file`` and ``download`` touch the weights on disk, and only ``download`` uses the
network — it is called by ``fpab setup-model`` alone, never by an analysis run. Every install is verified against
the recorded size and SHA-256; anything else is refused.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from .model import ModelError, sha256

RELEASE = "breath-model-v2"
FILENAME = "respiro-en-fpab-v2.safetensors"
WEIGHTS_SHA256 = "17dc2cb7bac027a2d5aa72649e575936730790785fe9d2edef6a82206dcf587d"
WEIGHTS_BYTES = 11_713_964
WEIGHTS_URL = f"https://github.com/alexeymohr/FinalPassAudioBook/releases/download/{RELEASE}/{FILENAME}"


def model_dir() -> Path:
    root = os.environ.get("FPAB_MODEL_DIR")
    base = Path(root) if root else Path.home() / ".cache" / "finalpass-audiobook"
    return base / "breath-respiro" / WEIGHTS_SHA256[:12]


def weights_path() -> Path:
    return model_dir() / FILENAME


def _verify(path: Path) -> None:
    size = path.stat().st_size
    if size != WEIGHTS_BYTES:
        raise ModelError(f"{path}: {size} bytes, expected {WEIGHTS_BYTES}")
    digest = sha256(path)
    if digest != WEIGHTS_SHA256:
        raise ModelError(f"{path}: SHA-256 {digest} does not match the recorded {WEIGHTS_SHA256}")


def _place(tmp: Path) -> Path:
    _verify(tmp)
    dest = weights_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, dest)
    return dest


def install_from_file(src: Path) -> Path:
    """Copy an already-downloaded weights file into place, after verifying it."""
    _verify(src)
    model_dir().mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=model_dir(), delete=False) as fh:
        tmp = Path(fh.name)
    try:
        shutil.copyfile(src, tmp)
        return _place(tmp)
    finally:
        tmp.unlink(missing_ok=True)                     # moved into place already, or a failed copy


def download() -> Path:
    """Fetch the weights from this repository's release and verify them."""
    model_dir().mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=model_dir(), delete=False) as fh:
        tmp = Path(fh.name)
    try:
        with open(tmp, "wb") as out, urllib.request.urlopen(WEIGHTS_URL, timeout=60) as r:  # noqa: S310 (pinned)
            got = 0
            while chunk := r.read(1 << 20):
                got += len(chunk)
                if got > WEIGHTS_BYTES:                 # never fill the disk with an unexpected file
                    raise ModelError(f"download is larger than the recorded {WEIGHTS_BYTES} bytes")
                out.write(chunk)
        return _place(tmp)
    finally:
        tmp.unlink(missing_ok=True)                     # the verified copy was moved into place already


def installed() -> bool:
    try:
        return weights_path().is_file()
    except OSError:                             # a model folder that cannot be looked at: not usable
        return False


def load():
    """The numpy breath model with the verified weights. No network, no PyTorch. The bytes that are hashed are the
    bytes that are parsed."""
    from .breath_np import BreathModel
    from .truncation_np import WeightsError, parse_safetensors

    if not installed():
        raise ModelError("breath model weights not installed — run `fpab setup-model`")
    path = weights_path()
    try:
        with open(path, "rb") as fh:
            size = os.fstat(fh.fileno()).st_size
            if size != WEIGHTS_BYTES:
                raise ModelError(f"{path}: {size} bytes, expected {WEIGHTS_BYTES}")
            raw = fh.read(WEIGHTS_BYTES + 1)
    except OSError as exc:
        raise ModelError(f"{path}: cannot be read ({exc})") from exc
    if len(raw) != WEIGHTS_BYTES:
        raise ModelError(f"{path}: {len(raw)} bytes, expected {WEIGHTS_BYTES}")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != WEIGHTS_SHA256:
        raise ModelError(f"{path}: SHA-256 {digest} does not match the recorded {WEIGHTS_SHA256}")
    try:
        return BreathModel(parse_safetensors(raw))
    except (WeightsError, ValueError, KeyError) as exc:
        raise ModelError(f"{path}: {exc}") from exc
