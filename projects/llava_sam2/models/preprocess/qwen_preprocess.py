


import numpy as np
from transformers import AutoProcessor

class QwenPrepocessor():
    def __init__(self, model_path, min_pixels=8*8*28*28, max_pixels=24*24*28*28) -> None:
        self.processor = AutoProcessor.from_pretrained(
            model_path,
        )
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def __call__(self, images=None, videos=None, text=None):
        return self.processor(images=images, videos=videos, text=text, min_pixels=self.min_pixels, max_pixels=self.max_pixels, fps=1)
