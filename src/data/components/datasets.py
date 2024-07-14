import torch
from torch.utils.data import Dataset
import numpy as np
from data.manifest import Manifest
from PIL import Image
from src.data.components.augmentor import Augmentor

# リソースの指定（CPU/GPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HeightDataset(Dataset):
    def __init__(self, manifests_path: str):
        self.manifests = Manifest.load_jsonp(manifests_path)
        self.augmentor = Augmentor()

    def __len__(self):
        return len(self.manifests)

    def __getitem__(self, index):
        # 全てのimageはchannel, height, widthのサイズ

        manifest = self.manifests[index]

        original = Image.open(manifest.original_image_path).convert(mode="L")
        pose = Image.open(manifest.pose_image_path).convert(mode="L")
        depth = Image.open(manifest.depth_image_path)
        mask = Image.open(manifest.mask_image_path)

        original, pose, depth, mask = self.augmentor.augment(
            original, depth, pose, mask
        )

        original = np.expand_dims(original, 0)
        pose = np.expand_dims(pose, 0)
        depth = np.expand_dims(depth, 0)
        mask = np.expand_dims(mask, 0)

        original = torch.tensor(original, dtype=torch.float).to(device)
        pose = torch.tensor(pose, dtype=torch.float).to(device)
        depth = torch.tensor(depth, dtype=torch.float).to(device)
        mask = torch.tensor(mask, dtype=torch.float).to(device)

        height = torch.tensor(manifest.height).to(device)
        return original, pose, depth, mask, height


# データローダーのサブプロセスの乱数seedが固定
def collent_fn(batch):
    original, pose, depth, mask, height = zip(*batch)
    # サイズを揃えるためにpadding
    max_height = max([o.size(1) for o in original])
    max_width = max([o.size(2) for o in original])
    original = [
        torch.nn.functional.pad(
            o, (0, max_width - o.size(2), 0, max_height - o.size(1))
        )
        for o in original
    ]
    pose = [
        torch.nn.functional.pad(
            p, (0, max_width - p.size(2), 0, max_height - p.size(1))
        )
        for p in pose
    ]
    depth = [
        torch.nn.functional.pad(
            d, (0, max_width - d.size(2), 0, max_height - d.size(1))
        )
        for d in depth
    ]
    mask = [
        torch.nn.functional.pad(
            m, (0, max_width - m.size(2), 0, max_height - m.size(1))
        )
        for m in mask
    ]

    height = torch.stack(height)
    return (
        torch.stack(original),
        torch.stack(pose),
        torch.stack(depth),
        torch.stack(mask),
        height,
    )
