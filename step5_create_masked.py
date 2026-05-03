"""
Step 5: Apply the block letter mask to the stippled image.
Dark mask regions (the letter) become white—stipples removed for the biased estimate.
"""

from __future__ import annotations

import numpy as np


def create_masked_stipple(
    stipple_img: np.ndarray,
    mask_img: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Remove stipples wherever the mask is dark (the letter area).

    Parameters
    ----------
    stipple_img : np.ndarray
        Stippled image, shape ``(H, W)``, values typically in ``[0, 1]``.
    mask_img : np.ndarray
        Block letter image: letter ≈ 0, background ≈ 1, same shape as ``stipple_img``.
    threshold : float
        Pixels with ``mask_img < threshold`` are treated as mask (cleared to white).

    Returns
    -------
    np.ndarray
        Same shape as inputs; stipples kept outside the letter, set to ``1.0`` inside.

    Raises
    ------
    ValueError
        If shapes differ or arrays are not 2D.
    """
    if stipple_img.ndim != 2 or mask_img.ndim != 2:
        raise ValueError(
            "stipple_img and mask_img must be 2D arrays; "
            f"got ndim={stipple_img.ndim} and {mask_img.ndim}."
        )
    if stipple_img.shape != mask_img.shape:
        raise ValueError(
            "Shape mismatch: stipple_img "
            f"{stipple_img.shape}, mask_img {mask_img.shape}."
        )

    stipple_f = np.asarray(stipple_img, dtype=np.float64)
    mask_f = np.asarray(mask_img, dtype=np.float64)
    masked = np.where(mask_f < threshold, 1.0, stipple_f)
    return masked.astype(stipple_img.dtype, copy=False)
