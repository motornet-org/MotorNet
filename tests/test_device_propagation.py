"""Tests for device propagation through .to() across the Environment -> Effector -> {Muscle, Skeleton} tree.

Environment, Effector, Muscle, and Skeleton each override .to() to track their own `self._device`
bookkeeping attribute (used elsewhere as `device=self.device` when constructing new tensors).
Calling .to(device) on a parent triggers PyTorch's internal nn.Module._apply() recursion into
children, which correctly moves already-registered parameters/buffers -- but _apply() calls each
child's `_apply`, not its public `.to()`, so a child's own `.to()` override (and the `self._device`
bookkeeping it updates) is never invoked unless `.to()` is called on that child directly. Nested
`self._device` attributes can therefore go stale relative to where the real tensors live, causing
device-mismatch errors when the module later constructs new tensors using `device=self.device`.

These tests require a non-CPU device (CUDA or MPS) to actually observe the bug, since moving
cpu -> cpu can't expose a staleness problem. They are skipped if neither is available.
"""

import pytest
import torch

from motornet.effector import ReluPointMass24
from motornet.environment import Environment


def _first_available_accelerator():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


DEVICE = _first_available_accelerator()
requires_accelerator = pytest.mark.skipif(
    DEVICE is None, reason="no non-CPU device (cuda/mps) available"
)


@requires_accelerator
class TestDevicePropagation:

    @pytest.fixture
    def env(self):
        return Environment(effector=ReluPointMass24())

    def test_effector_device_updates_after_env_to(self, env):
        env.to(DEVICE)
        assert env.effector.device.type == DEVICE

    def test_muscle_device_updates_after_env_to(self, env):
        env.to(DEVICE)
        assert env.effector.muscle.device.type == DEVICE

    def test_skeleton_device_updates_after_env_to(self, env):
        env.to(DEVICE)
        assert env.effector.skeleton.device.type == DEVICE

    def test_reset_after_env_to_does_not_raise(self, env):
        """End-to-end reproduction of the actual crash: reset() creates new tensors using the
        (potentially stale) device bookkeeping of effector/muscle/skeleton, which raises a
        device-mismatch RuntimeError against tensors that were genuinely moved by _apply()."""
        env.to(DEVICE)
        env.reset(options={"batch_size": 4, "deterministic": True})
