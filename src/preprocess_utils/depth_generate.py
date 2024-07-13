from transformers import pipeline
from PIL import Image
import cv2
from numpy.typing import NDArray


class DepthGenerate:
    def __init__(self):
        self.pipe = pipeline(
            task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf"
        )

    def generate(self, image_path: str, save_path: str):
        image = Image.open(image_path)
        depth: Image = self.pipe(image)["depth"]
        depth.save(save_path)
