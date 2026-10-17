"""Safe image loading, validation, EXIF correction, and format checks."""

import io
from pathlib import Path
from typing import BinaryIO, Optional, Tuple, Union

from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import settings

# Safe maximum pixel count to prevent decompression bomb attacks (approx 50 megapixels)
Image.MAX_IMAGE_PIXELS = 50_000_000

# Map PIL format strings to allowed standard formats
SUPPORTED_PIL_FORMATS = {"JPEG", "PNG", "WEBP"}


class ImageValidationError(Exception):
    """Base exception for all user-facing image validation failures."""
    pass


class ImageSizeError(ImageValidationError):
    """Raised when an image file exceeds allowed size limits."""
    pass


class ImageFormatError(ImageValidationError):
    """Raised when an image format is unsupported or invalid."""
    pass


class ImageDimensionError(ImageValidationError):
    """Raised when image dimensions are too small or unreasonably large."""
    pass


class ImageCorruptError(ImageValidationError):
    """Raised when image bytes are corrupted or cannot be decoded."""
    pass


def validate_and_load_image(
    image_input: Union[str, Path, bytes, BinaryIO],
    max_size_bytes: Optional[int] = None,
    min_dimension: Optional[int] = None,
    max_dimension: Optional[int] = None,
) -> Image.Image:
    """Safely validate, decode, EXIF-transpose, and convert an image to RGB.
    
    Args:
        image_input: File path, raw bytes, or file-like binary stream.
        max_size_bytes: Maximum allowed byte size (defaults to config: 10MB).
        min_dimension: Minimum allowed width/height in pixels (defaults to config: 50).
        max_dimension: Maximum allowed width/height in pixels (defaults to config: 10000).

    Returns:
        PIL.Image.Image in RGB mode.

    Raises:
        ImageSizeError: If byte size exceeds limits.
        ImageFormatError: If file format is not JPEG, PNG, or WebP.
        ImageDimensionError: If dimensions are out of bounds.
        ImageCorruptError: If image data is corrupted or cannot be parsed.
    """
    max_size_bytes = max_size_bytes or settings.max_image_file_size_bytes
    min_dimension = min_dimension or settings.min_image_dimension
    max_dimension = max_dimension or settings.max_image_dimension

    raw_bytes: bytes

    # 1. Read bytes and check size
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.is_file():
            raise ImageValidationError(f"Image file does not exist: {p}")
        file_size = p.stat().st_size
        if file_size > max_size_bytes:
            raise ImageSizeError(
                f"Image file size ({file_size / (1024 * 1024):.1f} MB) exceeds maximum allowed limit of "
                f"{max_size_bytes / (1024 * 1024):.0f} MB."
            )
        try:
            with open(p, "rb") as f:
                raw_bytes = f.read()
        except Exception as e:
            raise ImageCorruptError(f"Failed to read image file from disk: {e}") from e

    elif isinstance(image_input, bytes):
        raw_bytes = image_input
        if len(raw_bytes) > max_size_bytes:
            raise ImageSizeError(
                f"Image data size ({len(raw_bytes) / (1024 * 1024):.1f} MB) exceeds maximum allowed limit of "
                f"{max_size_bytes / (1024 * 1024):.0f} MB."
            )

    elif hasattr(image_input, "read"):
        try:
            # If stream has tell/seek or getbuffer (e.g. BytesIO / UploadedFile)
            if hasattr(image_input, "seek"):
                image_input.seek(0)
            raw_bytes = image_input.read()
            if hasattr(image_input, "seek"):
                image_input.seek(0)
        except Exception as e:
            raise ImageCorruptError(f"Failed to read from image stream: {e}") from e

        if len(raw_bytes) > max_size_bytes:
            raise ImageSizeError(
                f"Image size ({len(raw_bytes) / (1024 * 1024):.1f} MB) exceeds maximum allowed limit of "
                f"{max_size_bytes / (1024 * 1024):.0f} MB."
            )
    else:
        raise ImageValidationError("Unsupported image input type.")

    if not raw_bytes:
        raise ImageCorruptError("Image data is empty (0 bytes).")

    # 2. Decode with Pillow and verify format
    try:
        stream = io.BytesIO(raw_bytes)
        img = Image.open(stream)
        img.load()  # Force load pixel data to detect corruption early
    except UnidentifiedImageError:
        raise ImageFormatError("Unsupported file type. Please upload a valid JPEG, PNG, or WebP photo.")
    except Image.DecompressionBombError:
        raise ImageDimensionError("Image is too large and was rejected for security (decompression bomb protection).")
    except Exception as e:
        raise ImageCorruptError(f"Image data appears corrupted: {e}") from e

    # 3. Verify format matches supported types
    detected_format = (img.format or "").upper()
    if detected_format not in SUPPORTED_PIL_FORMATS:
        raise ImageFormatError(
            f"Image format '{detected_format}' is not supported. Please use JPEG, PNG, or WebP."
        )

    # 4. Dimension checks
    width, height = img.size
    if width < min_dimension or height < min_dimension:
        raise ImageDimensionError(
            f"Image dimensions ({width}x{height}) are too small. Minimum required dimension is {min_dimension}x{min_dimension} pixels."
        )
    if width > max_dimension or height > max_dimension:
        raise ImageDimensionError(
            f"Image dimensions ({width}x{height}) exceed maximum allowed dimension of {max_dimension} pixels."
        )

    # 5. Handle EXIF orientation safely
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        # If EXIF reading fails, continue with original orientation rather than failing
        pass

    # 6. Convert to standard RGB mode
    if img.mode != "RGB":
        try:
            # If RGBA, create white background to avoid dark transparent artifacts
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                rgba_img = img.convert("RGBA")
                background = Image.new("RGBA", rgba_img.size, (255, 255, 255, 255))
                alpha_composite = Image.alpha_composite(background, rgba_img)
                img = alpha_composite.convert("RGB")
            else:
                img = img.convert("RGB")
        except Exception as e:
            raise ImageCorruptError(f"Failed to convert image color mode ({img.mode}) to RGB: {e}") from e

    return img


def check_image_path(path: Union[str, Path]) -> Tuple[bool, Optional[str]]:
    """Quick validation check for catalog images on disk.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        validate_and_load_image(path)
        return True, None
    except ImageValidationError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error loading image: {e}"

