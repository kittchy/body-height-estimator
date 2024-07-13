import torch
from torch.utils.data import Dataset
import numpy as np
from data.manifest import Manifest
from PIL import Image

# リソースの指定（CPU/GPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HeightDataset(Dataset):
    def __init__(self, manifests_path: str):
        self.manifests = Manifest.load_jsonp(manifests_path)

    def __len__(self):
        return len(self.manifests)

    def __getitem__(self, index):
        # 全てのimageはchannel, height, widthのサイズ

        manifest = self.manifests[index]

        original = Image.open(manifest.original_image_path).convert(mode="L")
        original = np.expand_dims(np.array(original), 0) / 255.0  # 0~255 -> 0~1
        original = torch.tensor(original, dtype=torch.float).to(device)

        pose_image = Image.open(manifest.pose_image_path).convert(mode="L")
        pose_image = np.expand_dims(np.array(pose_image), 0) / 255.0  # 0~255 -> 0~1
        pose_image = torch.tensor(pose_image, dtype=torch.float).to(device)

        depth_image = Image.open(manifest.depth_image_path)
        depth_image = np.expand_dims(np.array(depth_image), 0) / 255.0  # 0~255 -> 0~1
        depth_image = torch.tensor(depth_image, dtype=torch.float).to(device)

        mask_image = Image.open(manifest.mask_image_path)
        mask_image = np.expand_dims(np.array(mask_image), 0) / 255.0  # 0~255 -> 0~1
        mask_image = torch.tensor(mask_image, dtype=torch.float).to(device)

        height = torch.tensor(manifest.height).to(device)
        return original, pose_image, depth_image, mask_image, height


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
