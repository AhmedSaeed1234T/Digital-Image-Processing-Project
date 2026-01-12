# Preprocessing module
from .preprocessing import (
    load_image,
    to_gray,
    apply_histogram_equalization,
    apply_color_histogram_equalization,
    show_histogram,
    preprocess_folder
)

__all__ = [
    'load_image',
    'to_gray',
    'apply_histogram_equalization',
    'apply_color_histogram_equalization',
    'show_histogram',
    'preprocess_folder'
]

