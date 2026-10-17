"""Unit tests for safe image loading and validation."""

import io
from pathlib import Path
import pytest
from PIL import Image

from app.image_utils import (
    ImageCorruptError,
    ImageDimensionError,
    ImageFormatError,
    ImageSizeError,
    ImageValidationError,
    validate_and_load_image,
)


def create_test_image_bytes(format="PNG", size=(100, 100), color=(255, 0, 0), mode="RGB") -> bytes:
    """Helper to generate in-memory image bytes."""
    img = Image.new(mode, size, color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def test_valid_png_loading():
    data = create_test_image_bytes(format="PNG", size=(120, 120))
    img = validate_and_load_image(data)
    assert img.mode == "RGB"
    assert img.size == (120, 120)


def test_valid_jpeg_loading():
    data = create_test_image_bytes(format="JPEG", size=(150, 150))
    img = validate_and_load_image(data)
    assert img.mode == "RGB"
    assert img.size == (150, 150)


def test_valid_webp_loading():
    data = create_test_image_bytes(format="WEBP", size=(200, 200))
    img = validate_and_load_image(data)
    assert img.mode == "RGB"
    assert img.size == (200, 200)


def test_rgba_converts_to_rgb():
    data = create_test_image_bytes(format="PNG", size=(100, 100), color=(255, 0, 0, 128), mode="RGBA")
    img = validate_and_load_image(data)
    assert img.mode == "RGB"


def test_corrupt_bytes_rejected():
    corrupt_data = b"\x89PNG\r\n\x1a\nCorruptInvalidImageDataHere1234567890"
    with pytest.raises(ImageCorruptError):
        validate_and_load_image(corrupt_data)


def test_empty_bytes_rejected():
    with pytest.raises(ImageCorruptError):
        validate_and_load_image(b"")


def test_unsupported_format_rejected():
    # GIF is not in allowed {JPEG, PNG, WEBP}
    data = create_test_image_bytes(format="GIF", size=(100, 100))
    with pytest.raises(ImageFormatError):
        validate_and_load_image(data)


def test_oversized_file_rejected():
    small_limit = 500  # 500 bytes limit
    data = create_test_image_bytes(format="PNG", size=(300, 300))
    assert len(data) > small_limit
    with pytest.raises(ImageSizeError):
        validate_and_load_image(data, max_size_bytes=small_limit)


def test_too_small_dimensions_rejected():
    data = create_test_image_bytes(format="PNG", size=(30, 30))
    with pytest.raises(ImageDimensionError):
        validate_and_load_image(data, min_dimension=50)


def test_too_large_dimensions_rejected():
    data = create_test_image_bytes(format="PNG", size=(200, 200))
    with pytest.raises(ImageDimensionError):
        validate_and_load_image(data, max_dimension=150)

