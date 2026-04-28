"""
Step 4: Create a block letter mask (e.g. "S") matching image dimensions.
Black letter (0.0) on white background (1.0) for use as a selection-bias mask.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_block_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try common bold system fonts; fall back to PIL default bitmap font."""
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def create_block_letter_s(
    height: int,
    width: int,
    letter: str = "S",
    font_size_ratio: float = 0.9,
) -> np.ndarray:
    """
    Render a centered block letter on a white canvas, same size as your image grid.

    Parameters
    ----------
    height, width : int
        Output array shape (height, width), matching ``gray_image.shape``.
    letter : str
        Single character to draw (default ``"S"``).
    font_size_ratio : float
        Font size as a fraction of ``min(height, width)`` (try 0.85–0.95).

    Returns
    -------
    np.ndarray
        2D array, shape ``(height, width)``, values in ``[0, 1]``:
        letter ≈ 0.0, background 1.0.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid dimensions: height={height}, width={width}")
    if not letter:
        raise ValueError("letter must be a non-empty string")

    # PIL uses (width, height)
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)

    font_size = int(font_size_ratio * min(height, width))
    font_size = max(font_size, 8)
    font = _load_block_font(font_size)

    text = letter[0]
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    else:
        left, top = 0, 0
        right, bottom = draw.textsize(text, font=font)

    tw, th = right - left, bottom - top
    x = (width - tw) // 2 - left
    y = (height - th) // 2 - top

    draw.text((x, y), text, fill=0, font=font)

    out = np.asarray(img, dtype=np.float32) / 255.0
    return out
