from rembg import remove
from PIL import Image


class MaskGenerate:
    def __init__(self):
        pass

    def generate(self, image_path: str, save_path: str):
        input = Image.open(image_path)
        output: Image = remove(input, only_mask=True)
        output.save(save_path)
