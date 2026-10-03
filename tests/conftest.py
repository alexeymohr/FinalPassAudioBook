"""Every test starts from the same synthetic noise, whatever ran before it (no order dependence), and sees no
installed model (as on CI) unless it is marked real_models."""
from __future__ import annotations

import numpy as np
import pytest

import synth


@pytest.fixture(autouse=True)
def _fresh_synthetic_noise():
    synth.RNG.bit_generator.state = np.random.default_rng(synth.SEED).bit_generator.state
    yield


@pytest.fixture(autouse=True)
def _no_installed_models(request, tmp_path_factory, monkeypatch):  # noqa: ANN001
    """A local run tests what CI tests: no model weights, unless a test asks for the ones installed here."""
    if request.node.get_closest_marker("real_models") is None:
        monkeypatch.setenv("FPAB_MODEL_DIR", str(tmp_path_factory.mktemp("no-models")))
    yield
