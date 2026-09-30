"""The truncated-word model: install the weights once, load them offline.

``load`` runs the model with our numpy port (truncation_np; no PyTorch). The
audited torch code is kept as the reference: ``load_reference`` needs the
``truncation`` extra and is used to hold the port to it.

Only ``install_from_file`` and ``download`` touch the weights on disk, and only
``download`` uses the network — it is called by ``fpab setup-model`` alone,
never by an analysis run. Every install is verified against the size and
SHA-256 recorded when the model was audited; anything else is refused.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

REPO = "mythicinfinity/speech-truncation-detection-12M"
REVISION = "75ae15a354568c179c910a833129f68e3a845218"
WEIGHTS_SHA256 = "5e150a897364f4afdfb557584df9607f4217ea99e08b74186ffe919f67caa0c0"
WEIGHTS_BYTES = 51_820_912
WEIGHTS_URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/model.safetensors"
VENDOR = Path(__file__).parent / "vendor" / "speech_truncation"


class ModelError(RuntimeError):
    """The model cannot be installed or loaded."""


def model_dir() -> Path:
    root = os.environ.get("FPAB_MODEL_DIR")
    base = Path(root) if root else Path.home() / ".cache" / "finalpass-audiobook"
    return base / "speech-truncation-12M" / REVISION[:12]


def weights_path() -> Path:
    return model_dir() / "model.safetensors"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(path: Path) -> None:
    size = path.stat().st_size
    if size != WEIGHTS_BYTES:
        raise ModelError(f"{path}: {size} bytes, expected {WEIGHTS_BYTES}")
    digest = sha256(path)
    if digest != WEIGHTS_SHA256:
        raise ModelError(f"{path}: SHA-256 {digest} does not match the audited {WEIGHTS_SHA256}")


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
    """Fetch the weights from the pinned revision and verify them."""
    model_dir().mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=model_dir(), delete=False) as fh:
        tmp = Path(fh.name)
    try:
        with open(tmp, "wb") as out, urllib.request.urlopen(WEIGHTS_URL, timeout=60) as r:  # noqa: S310 (pinned)
            got = 0
            while chunk := r.read(1 << 20):
                got += len(chunk)
                if got > WEIGHTS_BYTES:                 # never fill the disk with an unexpected file
                    raise ModelError(f"download is larger than the audited {WEIGHTS_BYTES} bytes")
                out.write(chunk)
        return _place(tmp)
    finally:
        tmp.unlink(missing_ok=True)                     # the verified copy was moved into place already


def installed() -> bool:
    return weights_path().is_file()


def load():
    """The numpy model with the verified weights. No network, no PyTorch. The bytes that are
    hashed are the bytes that are parsed (no re-read between the check and the use)."""
    from .truncation_np import WeightsError, build_from_bytes

    if not installed():
        raise ModelError("model weights not installed — run `fpab setup-model`")
    path = weights_path()
    try:
        size = path.stat().st_size                  # never read an unexpected file into memory
        if size != WEIGHTS_BYTES:
            raise ModelError(f"{path}: {size} bytes, expected {WEIGHTS_BYTES}")
        raw = path.read_bytes()
    except OSError as exc:
        raise ModelError(f"{path}: cannot be read ({exc})") from exc
    if len(raw) != WEIGHTS_BYTES:
        raise ModelError(f"{path}: {len(raw)} bytes, expected {WEIGHTS_BYTES}")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != WEIGHTS_SHA256:
        raise ModelError(f"{path}: SHA-256 {digest} does not match the audited {WEIGHTS_SHA256}")
    try:
        return build_from_bytes(raw, VENDOR / "config.json")
    except (WeightsError, ValueError, KeyError) as exc:
        raise ModelError(f"{path}: {exc}") from exc


def load_reference():
    """The audited torch model (needs the `truncation` extra): the reference for the numpy port."""
    if not installed():
        raise ModelError("model weights not installed — run `fpab setup-model`")
    _verify(weights_path())
    try:
        from safetensors.torch import load_file

        from .vendor.speech_truncation import SpeechTruncationDetectionConfig, SpeechTruncationDetectionModel
    except ImportError as exc:
        raise ModelError(f"truncation extra not installed ({exc.name}) — "
                         "install finalpass-audiobook[truncation]") from exc
    cfg = SpeechTruncationDetectionConfig(**json.loads((VENDOR / "config.json").read_text(encoding="utf-8")))
    model = SpeechTruncationDetectionModel(cfg)
    model.load_state_dict(load_file(str(weights_path()), device="cpu"), strict=True)
    model.eval()
    return model
