


import numpy as np
from transformers import AutoProcessor

class QwenPrepocessor():
    def __init__(self, model_path, min_pixels=8*8*28*28, max_pixels=24*24*28*28, video_min_pixels=8*8*3*28*28, video_max_pixels=24*24*3*28*28) -> None:
        self.processor = AutoProcessor.from_pretrained(
            model_path,
        )
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.video_min_pixels = video_min_pixels
        self.video_max_pixels = video_max_pixels

    def __call__(self, images=None, videos=None, text=None):
        if images is not None:
            return self.processor(images=images, videos=None, text=text, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        elif videos is not None:
            return self.processor(images=None, videos=videos, text=text, min_pixels=self.video_min_pixels, max_pixels=self.video_max_pixels, fps=1)
        else:
            raise ValueError("images or videos must be provided")