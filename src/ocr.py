
"""OCR utilities for extracting text from uploaded images."""

from PIL import Image, ImageOps
import pytesseract


def extract_text_from_image(image_file) -> str:
    """
    Extract text from an uploaded image using Tesseract OCR.

    Args:
        image_file: Uploaded image file or image file-like object.

    Returns:
        Extracted text as a string.
    """

    # Open the uploaded image
    image = Image.open(image_file)

    # Convert image to grayscale for better OCR processing
    image = ImageOps.grayscale(image)

    # Improve image contrast
    image = ImageOps.autocontrast(image)

    # Extract text using Tesseract OCR
    extracted_text = pytesseract.image_to_string(
        image,
        lang="eng",
        config="--psm 6"
    )

    return extracted_text.strip()
