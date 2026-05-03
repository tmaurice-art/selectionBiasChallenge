"""
Assemble the four-panel statistics meme (Reality, Model, Selection Bias, Estimate).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def create_statistics_meme(
    original_img: np.ndarray,
    stipple_img: np.ndarray,
    block_letter_img: np.ndarray,
    masked_stipple_img: np.ndarray,
    output_path: str,
    dpi: int = 150,
    background_color: str = "white",
) -> None:
    """
    Save a 1×4 meme figure with panel labels.

    Parameters
    ----------
    original_img, stipple_img, block_letter_img, masked_stipple_img : np.ndarray
        2D grayscale arrays in ``[0, 1]``, all the same shape.
    output_path : str
        Path for the PNG (e.g. ``"my_statistics_meme.png"``).
    dpi : int
        Resolution for ``savefig``.
    background_color : str
        Matplotlib color for the figure background.
    """
    imgs = (original_img, stipple_img, block_letter_img, masked_stipple_img)
    for i, arr in enumerate(imgs):
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array at index {i}, got ndim={arr.ndim}")
    shape0 = imgs[0].shape
    if not all(a.shape == shape0 for a in imgs[1:]):
        raise ValueError(
            "All images must have the same shape; "
            f"got {[a.shape for a in imgs]}"
        )

    titles = ("Reality", "Your Model", "Selection Bias", "Estimate")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    fig.patch.set_facecolor(background_color)

    for ax, img, title in zip(axes, imgs, titles, strict=True):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_axis_off()
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=background_color)
    plt.close(fig)
