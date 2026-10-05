"""Tests for utility functions in comfyui-optional-asset-loader."""

import os
import tempfile
import numpy as np
import pytest
from PIL import Image

from utils import (
    create_blank_image_tensor,
    create_blank_mask_tensor,
    is_fallback_asset,
    load_image_to_tensor,
    parse_fallback_patterns,
    resolve_file_path,
)


def test_parse_fallback_patterns():
    patterns = parse_fallback_patterns("blank_start.png, NONE, Null , *empty*")
    assert "blank_start.png" in patterns
    assert "none" in patterns
    assert "null" in patterns
    assert "*empty*" in patterns


def test_is_fallback_asset_defaults():
    # Empty / None
    assert is_fallback_asset(None) is True
    assert is_fallback_asset("") is True
    assert is_fallback_asset("   ") is True
    assert is_fallback_asset("''") is True

    # Built-in fallback tokens
    assert is_fallback_asset("blank_start.png") is True
    assert is_fallback_asset("BLANK_START.PNG") is True
    assert is_fallback_asset("/some/path/to/blank_start.png") is True
    assert is_fallback_asset("blank_start.mp4") is True
    assert is_fallback_asset("none") is True
    assert is_fallback_asset("null") is True
    assert is_fallback_asset("empty") is True

    # Real filenames should not be fallback
    assert is_fallback_asset("my_character.png") is False
    assert is_fallback_asset("video_render_01.mp4") is False
    assert is_fallback_asset("/datapool/user_uploads/cat.jpg") is False


def test_is_fallback_asset_wildcard():
    custom_patterns = "*placeholder*, *dummy*, none"
    assert is_fallback_asset("test_placeholder_image.png", custom_patterns) is True
    assert is_fallback_asset("my_dummy_file.mp4", custom_patterns) is True
    assert is_fallback_asset("real_photo.png", custom_patterns) is False


def test_resolve_file_path_fallback():
    # Fallback paths should immediately return None
    assert resolve_file_path("blank_start.png") is None
    assert resolve_file_path("") is None
    assert resolve_file_path(None) is None


def test_resolve_file_path_real_file():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(b"dummy")
        tmp_path = tmp.name

    try:
        resolved = resolve_file_path(tmp_path)
        assert resolved == os.path.abspath(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_resolve_file_path_missing_file():
    assert resolve_file_path("non_existent_image_12345.png") is None


def test_blank_image_and_mask_tensors():
    img_black = create_blank_image_tensor(width=128, height=64, color="black", batch_size=2)
    assert img_black.shape == (2, 64, 128, 3)
    assert np.all(np.array(img_black) == 0.0)

    img_white = create_blank_image_tensor(width=64, height=64, color="white", batch_size=1)
    assert img_white.shape == (1, 64, 64, 3)
    assert np.all(np.array(img_white) == 1.0)

    mask = create_blank_mask_tensor(width=128, height=64, batch_size=2)
    assert mask.shape == (2, 64, 128)
    assert np.all(np.array(mask) == 0.0)


def test_load_image_to_tensor_rgb():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img = Image.new("RGB", (100, 50), color=(255, 0, 0))
        img.save(tmp.name)
        tmp_path = tmp.name

    try:
        image_tensor, mask_tensor = load_image_to_tensor(tmp_path)
        assert image_tensor.shape == (1, 50, 100, 3)
        assert mask_tensor.shape == (1, 50, 100)
        # Check red pixel normalized: [1.0, 0.0, 0.0]
        np_img = np.array(image_tensor)
        assert np.isclose(np_img[0, 0, 0, 0], 1.0)
        assert np.isclose(np_img[0, 0, 0, 1], 0.0)
        assert np.isclose(np_img[0, 0, 0, 2], 0.0)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_load_image_to_tensor_rgba():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        # 50% transparent image
        img = Image.new("RGBA", (80, 40), color=(0, 255, 0, 128))
        img.save(tmp.name)
        tmp_path = tmp.name

    try:
        image_tensor, mask_tensor = load_image_to_tensor(tmp_path)
        assert image_tensor.shape == (1, 40, 80, 3)
        assert mask_tensor.shape == (1, 40, 80)
        # ComfyUI mask: 1.0 - alpha/255 -> 1.0 - ~0.5 = ~0.5
        np_mask = np.array(mask_tensor)
        assert np.isclose(np_mask[0, 0, 0], 1.0 - (128 / 255.0), atol=0.02)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
