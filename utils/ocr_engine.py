from functools import lru_cache
import json

import numpy as np


# Hindi + English OCR
OCR_LANGUAGE = "hi"


@lru_cache(maxsize=1)
def get_ocr_engine():
    """
    Create and cache the PaddleOCR engine.

    lang='hi' uses PaddleOCR's Hindi/multilingual recognition model.
    The model is intended to recognize Devanagari text and can handle
    English characters appearing in the document as well.
    """
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang=OCR_LANGUAGE,

        # Disable optional document-processing components.
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def _to_plain_dict(item):
    """
    Convert a PaddleOCR prediction object into a normal Python dict.
    """

    if isinstance(item, dict):
        return item

    value = getattr(item, "json", None)

    if callable(value):
        value = value()

    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

    return {}


def extract_text(image):
    """
    Run Hindi + English OCR on a PIL image.

    Returns:
        {
            "text": "...",
            "lines": [...],
            "scores": [...],
            "mean_confidence": ...
        }
    """

    if image is None:
        raise ValueError("Image cannot be None.")

    # PIL -> NumPy
    image_array = np.asarray(
        image.convert("RGB")
    )

    # Run OCR
    predictions = get_ocr_engine().predict(
        image_array
    )

    all_texts = []
    all_scores = []

    for prediction in predictions or []:

        payload = _to_plain_dict(prediction)

        result = payload.get(
            "res",
            payload
        )

        texts = result.get(
            "rec_texts",
            []
        ) or []

        scores = result.get(
            "rec_scores",
            []
        ) or []

        # Extract recognized text
        for text in texts:

            text = str(text).strip()

            if text:
                all_texts.append(text)

        # Extract confidence scores
        for score in scores:

            try:
                all_scores.append(
                    float(score)
                )

            except (
                TypeError,
                ValueError
            ):
                continue

    # Keep Unicode unchanged.
    #
    # Example:
    #
    # This is an OCR system.
    # यह एक ओसीआर सिस्टम है।

    final_text = "\n".join(
        all_texts
    )

    mean_confidence = None

    if all_scores:
        mean_confidence = (
            sum(all_scores)
            / len(all_scores)
        )

    return {
        "text": final_text,
        "lines": all_texts,
        "scores": all_scores,
        "mean_confidence": mean_confidence,
    }