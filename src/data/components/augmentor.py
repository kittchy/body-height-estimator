from typing import ChainMap
from PIL import ImageOps, Image
from numpy.random import random
from numpy.typing import NDArray
import numpy as np


class Augmentor:
    def __init__(
        self,
        rescale=1.0 / 255,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=180,
        slide_range=0.2,
        random_mask=True,
        mask_rate=0.2,
        apply_rate=0.4,
    ):
        self.rescale = rescale
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        assert 0 <= rotation_range <= 180
        self.rotation_range = rotation_range
        self.slide_range = slide_range
        self.random_mask = random_mask
        self.mask_rate = mask_rate
        self.apply_rate = apply_rate

    def apply_random_mask(self, image: NDArray) -> NDArray:
        if np.random.rand() < self.mask_rate:
            ones = np.ones(image.shape)
            h, w = image.shape
            h_start = np.random.randint(0, h)
            h_range = int(np.random.randint(0, h - h_start) * self.apply_rate)
            w_start = np.random.randint(0, w)
            w_range = int(np.random.randint(0, w - w_start) * self.apply_rate)

            ones[h_start : h_start + h_range, w_start : w_start + w_range] = np.zeros(
                (h_range, w_range), dtype=float
            )
            image = image * ones
        return image

    def augment(
        self, original: Image, depth: Image, pose: Image, mask: Image
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        # 左右反転
        if self.horizontal_flip and np.random.rand() < 0.5:
            original = ImageOps.mirror(original)
            depth = ImageOps.mirror(depth)
            pose = ImageOps.mirror(pose)
            mask = ImageOps.mirror(mask)
        # 上下反転
        if self.vertical_flip and np.random.rand() < 0.5:
            original = ImageOps.flip(original)
            depth = ImageOps.flip(depth)
            pose = ImageOps.flip(pose)
            mask = ImageOps.flip(mask)

        if self.rotation_range > 0 and np.random.rand() < self.apply_rate:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            original = original.rotate(angle, expand=True)
            depth = depth.rotate(angle, expand=True)
            pose = pose.rotate(angle, expand=True)
            mask = mask.rotate(angle, expand=True)

        # zoom
        if self.zoom_range > 0 and np.random.rand() < self.apply_rate:
            zoom = np.random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
            weight, height = original.size
            weight = int(weight * zoom)
            height = int(height * zoom)
            original = original.resize((weight, height))
            depth = depth.resize((weight, height))
            pose = pose.resize((weight, height))
            mask = mask.resize((weight, height))
        # slide
        if self.slide_range > 0 and np.random.rand() < self.apply_rate:
            slide = np.random.uniform(-self.slide_range, self.slide_range)
            w, h = original.size
            original = original.rotate(
                0, translate=(w * self.slide_range, h * self.slide_range)
            )
            depth = depth.rotate(
                0, translate=(w * self.slide_range, h * self.slide_range)
            )
            pose = pose.rotate(
                0, translate=(w * self.slide_range, h * self.slide_range)
            )
            mask = mask.rotate(
                0, translate=(w * self.slide_range, h * self.slide_range)
            )

        # convert NDArray
        original_np = np.array(original)
        depth_np = np.array(depth)
        pose_np = np.array(pose)
        mask_np = np.array(mask)

        if self.random_mask and np.random.rand() < self.mask_rate:
            original_np = self.apply_random_mask(original_np)
            depth_np = self.apply_random_mask(depth_np)
            pose_np = self.apply_random_mask(pose_np)
            mask_np = self.apply_random_mask(mask_np)

        return (
            original_np * self.rescale,
            depth_np * self.rescale,
            pose_np * self.rescale,
            mask_np * self.rescale,
        )


if __name__ == "__main__":
    augmentor = Augmentor()
    random_image = Image.open("data/Dump/original/503-090_Caitlin_L1.jpg")
    original_np, depth_np, pose_np, mask_np = augmentor.augment(
        random_image, random_image, random_image, random_image
    )
