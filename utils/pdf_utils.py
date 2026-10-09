from io import BytesIO

from PIL import Image
import pymupdf


def pdf_to_images(
    pdf_bytes: bytes,
    dpi: int = 200
):
    """
    Convert every PDF page into a PIL RGB image.

    Returns:

    [
        {
            "page": 1,
            "image": PIL.Image
        },
        ...
    ]
    """

    if not pdf_bytes:
        raise ValueError(
            "The PDF file is empty."
        )

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    # PDF default resolution = 72 DPI
    scale = dpi / 72

    matrix = pymupdf.Matrix(
        scale,
        scale
    )

    try:

        for page_number, page in enumerate(
            document,
            start=1
        ):

            pixmap = page.get_pixmap(
                matrix=matrix,
                alpha=False
            )

            image_bytes = pixmap.tobytes(
                "png"
            )

            image = Image.open(
                BytesIO(image_bytes)
            ).convert("RGB")

            pages.append(
                {
                    "page": page_number,
                    "image": image,
                }
            )

    finally:

        document.close()

    return pages