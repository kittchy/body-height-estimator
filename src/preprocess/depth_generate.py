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
        depth: NDArray = self.pipe(image)["depth"]
        depth = cv2.cvtColor(depth, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, depth)
