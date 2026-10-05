"""Tests verifying tensor format compatibility when torch is simulated or present."""

from unittest.mock import MagicMock
import numpy as np
import pytest
import utils


def test_torch_present_or_fallback():
    # Test tensor creation
    img = utils.create_blank_image_tensor(64, 64, color="black", batch_size=1)
    mask = utils.create_blank_mask_tensor(64, 64, batch_size=1)

    assert hasattr(img, "shape")
    assert hasattr(mask, "shape")
    assert img.shape == (1, 64, 64, 3)
    assert mask.shape == (1, 64, 64)


def test_mock_torch_conversions(monkeypatch):
    """Simulate torch availability to ensure torch.from_numpy and cat work properly."""
    class FakeTensor:
        def __init__(self, arr):
            self.arr = np.array(arr)
            self.shape = self.arr.shape
            self.dtype = "torch.float32"

        def unsqueeze(self, dim):
            return FakeTensor(np.expand_dims(self.arr, axis=dim))

        def __rsub__(self, other):
            return FakeTensor(other - self.arr)

    class FakeTorch:
        float32 = "torch.float32"

        @staticmethod
        def from_numpy(arr):
            return FakeTensor(arr)

        @staticmethod
        def zeros(shape, dtype=None, device=None):
            return FakeTensor(np.zeros(shape, dtype=np.float32))

        @staticmethod
        def cat(tensors, dim=0):
            arrays = [t.arr for t in tensors]
            return FakeTensor(np.concatenate(arrays, axis=dim))

    monkeypatch.setattr(utils, "torch", FakeTorch)

    img = utils.create_blank_image_tensor(32, 16, color="white", batch_size=2)
    assert isinstance(img, FakeTensor)
    assert img.shape == (2, 16, 32, 3)
    assert np.all(img.arr == 1.0)
