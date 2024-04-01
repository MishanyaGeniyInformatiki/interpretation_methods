from matplotlib import pyplot as plt
import colorcet as cc
import numpy as np


def normalize(mask, vmin=None, vmax=None, percentile=99):
    if vmax is None:
        vmax = np.percentile(mask, percentile)
    if vmin is None:
        vmin = np.min(mask)
    return (mask - vmin) / (vmax - vmin + 1e-10)


def draw_input_image(image, title='', axis=None):
    axis.imshow(image, cmap='gray')
    axis.set_title(title)


def show_mask_my(mask, title='', axis=None):
    mask_norm = normalize(mask)
    axis.imshow(mask_norm, cmap='magma', alpha=1.0, vmin=0, vmax=1, interpolation='lanczos')
    axis.set_title(title)


def show_mask_on_image_my(image, mask, title='', cmap=cc.cm.bmy, k=0.7, axis=None):
    """
    k: k in [0, 1], - порог значений маски. Значения маски в диапазоне [0, 1]
    """
    image = image.permute(1, 2, 0)
    axis.imshow(image, cmap='gray', alpha=1.0)
    mask_norm = normalize(mask)
    mask_ = mask_norm.copy()
    mask_[mask_ <= k] = 0  # устанавливаем остальные значения в черный

    axis.imshow(mask_, cmap='magma', alpha=0.4, vmin=0, vmax=1, interpolation='lanczos')  # alpha=0.4 - процент прозрачности маски
    axis.set_title(title)
    plt.show()


def save_explanation(explanation, path):
    with open(path, "wb") as file:
        explanation.dump(file)
    file.close()
