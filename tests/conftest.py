"""Every test starts from the same synthetic noise, whatever ran before it (no order dependence)."""
from __future__ import annotations

import numpy as np
import pytest

import synth


@pytest.fixture(autouse=True)
def _fresh_synthetic_noise():
    synth.RNG.bit_generator.state = np.random.default_rng(synth.SEED).bit_generator.state
    yield
