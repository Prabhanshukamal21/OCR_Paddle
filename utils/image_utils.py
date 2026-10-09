from io import BytesIO

from PIL import Image, ImageOps


MAX_IMAGE_BYTES = 15 * 1024 * 1024
ALLOWED_FORMATS = {"PNG", "JPEG", "BMP", "WEBP"}


def load_uploaded_image(file_bytes: bytes) -> Image.Image:
    """Validate uploaded image bytes and return a correctly oriented RGB image."""
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")
    if len(file_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Image is larger than 15 MB.")

    try:
        with Image.open(BytesIO(file_bytes)) as image:
            if image.format not in ALLOWED_FORMATS:
                raise ValueError("Unsupported image format.")
            image.verify()

        with Image.open(BytesIO(file_bytes)) as image:
            image = ImageOps.exif_transpose(image)
            return image.convert("RGB")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("The file is not a valid supported image.") from exc
