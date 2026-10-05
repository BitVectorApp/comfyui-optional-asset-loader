"""Unit tests for ComfyUI custom nodes in comfyui-optional-asset-loader."""

import os
import tempfile
import numpy as np
import pytest
from PIL import Image

from nodes import (
    LoadImageOptional,
    LoadVideoOptional,
    NullableConstant,
    OptionalDimensionFilter,
    OptionalImageGate,
    OptionalVideoGate,
)


def test_load_image_optional_fallback_none():
    node = LoadImageOptional()
    # Empty string
    img, mask, has_image, path = node.load(image_path="", missing_behavior="Return None")
    assert img is None
    assert mask is None
    assert has_image is False
    assert path == ""

    # Fallback string
    img, mask, has_image, path = node.load(
        image_path="blank_start.png", missing_behavior="Return None"
    )
    assert img is None
    assert mask is None
    assert has_image is False
    assert path == ""

    # Non-existent file
    img, mask, has_image, path = node.load(
        image_path="does_not_exist_file.png", missing_behavior="Return None"
    )
    assert img is None
    assert mask is None
    assert has_image is False
    assert path == ""


def test_load_image_optional_fallback_blank_tensor():
    node = LoadImageOptional()
    img, mask, has_image, path = node.load(
        image_path="blank_start.png",
        missing_behavior="Return Blank Tensor",
        blank_width=256,
        blank_height=128,
        blank_color="black",
    )
    assert img is not None
    assert mask is not None
    assert has_image is False
    assert img.shape == (1, 128, 256, 3)
    assert mask.shape == (1, 128, 256)


def test_load_image_optional_raise_error():
    node = LoadImageOptional()
    with pytest.raises(FileNotFoundError):
        node.load(image_path="blank_start.png", missing_behavior="Raise Error")

    with pytest.raises(FileNotFoundError):
        node.load(image_path="missing_file.png", missing_behavior="Raise Error")


def test_load_image_optional_valid_file():
    node = LoadImageOptional()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pil_img = Image.new("RGB", (320, 240), color=(10, 20, 30))
        pil_img.save(tmp.name)
        tmp_path = tmp.name

    try:
        img, mask, has_image, path = node.load(image_path=tmp_path)
        assert img is not None
        assert mask is not None
        assert has_image is True
        assert path == os.path.abspath(tmp_path)
        assert img.shape == (1, 240, 320, 3)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_load_video_optional_fallback():
    node = LoadVideoOptional()
    # Return None
    frames, audio, has_video, count, fps, path = node.load(
        video_path="blank_start.mp4", missing_behavior="Return None"
    )
    assert frames is None
    assert audio is None
    assert has_video is False
    assert count == 0
    assert fps == 0.0
    assert path == ""

    # Return Blank Tensor
    frames, audio, has_video, count, fps, path = node.load(
        video_path="",
        missing_behavior="Return Blank Tensor",
        blank_width=128,
        blank_height=64,
        blank_frames=2,
    )
    assert frames is not None
    assert audio is None
    assert has_video is False
    assert frames.shape == (2, 64, 128, 3)


def test_optional_dimension_filter():
    node = OptionalDimensionFilter()

    # None input
    img_out, passed, w, h = node.filter(
        dimension_target="width", comparison="greater_than", threshold=91, image=None
    )
    assert img_out is None
    assert passed is False
    assert w == 0
    assert h == 0

    # Dummy image with width = 91 (the Graydient fallback width!)
    dummy_fallback = np.zeros((1, 91, 91, 3), dtype=np.float32)
    img_out, passed, w, h = node.filter(
        dimension_target="width",
        comparison="greater_than",
        threshold=91,
        fallback_behavior="Return None",
        image=dummy_fallback,
    )
    assert img_out is None
    assert passed is False
    assert w == 91
    assert h == 91

    # Real user image with width = 1024
    real_img = np.zeros((1, 512, 1024, 3), dtype=np.float32)
    img_out, passed, w, h = node.filter(
        dimension_target="width",
        comparison="greater_than",
        threshold=91,
        fallback_behavior="Return None",
        image=real_img,
    )
    assert img_out is real_img
    assert passed is True
    assert w == 1024
    assert h == 512

    # Dimension filter returning blank tensor
    img_out, passed, w, h = node.filter(
        dimension_target="width",
        comparison="greater_than",
        threshold=91,
        fallback_behavior="Return Blank Tensor",
        blank_width=256,
        blank_height=256,
        image=dummy_fallback,
    )
    assert img_out is not None
    assert passed is False
    assert img_out.shape == (1, 256, 256, 3)


def test_optional_image_gate():
    node = OptionalImageGate()
    sample_img = np.zeros((1, 100, 100, 3), dtype=np.float32)
    sample_mask = np.zeros((1, 100, 100), dtype=np.float32)

    # Active
    img, mask, has_image = node.gate(is_active=True, image=sample_img, mask=sample_mask)
    assert img is sample_img
    assert mask is sample_mask
    assert has_image is True

    # Inactive - Return None
    img, mask, has_image = node.gate(
        is_active=False, fallback_behavior="Return None", image=sample_img, mask=sample_mask
    )
    assert img is None
    assert mask is None
    assert has_image is False

    # Inactive - Return Blank
    img, mask, has_image = node.gate(
        is_active=False,
        fallback_behavior="Return Blank Tensor",
        blank_width=64,
        blank_height=64,
        image=sample_img,
    )
    assert img is not None
    assert mask is not None
    assert img.shape == (1, 64, 64, 3)
    assert has_image is False


def test_optional_video_gate():
    node = OptionalVideoGate()
    sample_vid = np.zeros((5, 100, 100, 3), dtype=np.float32)
    sample_audio = {"waveform": None, "sample_rate": 44100}

    # Active
    vid, aud, has_video = node.gate(is_active=True, video_frames=sample_vid, audio=sample_audio)
    assert vid is sample_vid
    assert aud is sample_audio
    assert has_video is True

    # Inactive
    vid, aud, has_video = node.gate(
        is_active=False, fallback_behavior="Return None", video_frames=sample_vid, audio=sample_audio
    )
    assert vid is None
    assert aud is None
    assert has_video is False


def test_nullable_constant():
    node = NullableConstant()
    (res,) = node.get_none()
    assert res is None
