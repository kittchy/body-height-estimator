# use https://www.kaggle.com/datasets/burnoutminer/heights-and-weights-dataset
from dataclasses import dataclass, field
from pathlib import Path
from dataclass_argparse import TypedNamespace
import shutil
import random
from pydantic import BaseModel

random.seed(0)


@dataclass
class Args(TypedNamespace):
    data_path: str = field(default="data/HWData/", metadata={"help": "database path"})
    dump_path: str = field(default="data/Dump", metadata={"help": "output file"})
    manifest_path: str = field(
        default="data/Manifest", metadata={"help": "manifest file"}
    )


class Manifest(BaseModel):
    original_image_path: str
    depth_image_path: str
    rmbg_image_path: str
    openpose_image_path: str
    height: float


def feet_to_cm(feet_inch: str):
    """
    Convert feet-inch to cm
    feet_inch: str : feet-inch value (e.g. 4' 11)
    """

    feet, inch = feet_inch.split("'")
    feet = int(feet)
    inch = int(inch)
    return feet * 30.48 + inch * 2.54


def preprocess(args: Args):
    datapath = Path(args.data_path)
    dump_path = Path(args.dump_path)
    manifest_path = Path(args.manifest_path)
    assert datapath.exists(), f"Path {datapath} does not exist"
    csv_file = datapath / "Output_data.csv"
    assert csv_file.exists(), f"File {csv_file} does not exist"

    # prepare dump and manifest
    if dump_path.exists():
        shutil.rmtree(str(dump_path))
    dump_path.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        shutil.rmtree(str(manifest_path))
    manifest_path.mkdir(parents=True, exist_ok=True)

    manifests: list[Manifest] = []

    with open(str(csv_file), "r") as f:
        metadata = f.readlines()
    for line in metadata[1:]:
        # extract metadata
        _, _, filename, h_w = line.rstrip().split(",")
        image_path = datapath / filename
        # h_w : " 4' 11"" 110 lbs."
        _, h_str, _, _, _ = h_w.split('"')
        h_cm = feet_to_cm(h_str)

        # copy image to dump
        original_image_path = dump_path / filename
        shutil.copy(str(image_path), str(original_image_path))
        # openpose image
        openpose_image_path = dump_path / filename
        # rmbg image
        rmgb_image_path = dump_path / filename
        # depth image
        depth_image_path = dump_path / filename

        manifests.append(
            Manifest(
                original_image_path=str(original_image_path),
                depth_image_path=str(depth_image_path),
                rmbg_image_path=str(rmgb_image_path),
                openpose_image_path=str(openpose_image_path),
                height=h_cm,
            )
        )
    random.shuffle(manifests)

    # output manifests
    train = manifests[: int(0.8 * len(manifests))]
    valid = manifests[int(0.8 * len(manifests)) : int(0.9 * len(manifests))]
    test = manifests[int(0.9 * len(manifests)) :]

    with open(str(manifest_path / "train.jsonp"), "w") as f:
        for manifest in train:
            f.write(manifest.model_dump_json() + "\n")
    with open(str(manifest_path / "valid.jsonp"), "w") as f:
        for manifest in valid:
            f.write(manifest.model_dump_json() + "\n")
    with open(str(manifest_path / "test.jsonp"), "w") as f:
        for manifest in test:
            f.write(manifest.model_dump_json() + "\n")


if __name__ == "__main__":
    parser = Args.get_parser_grouped_by_parents()
    args = parser.parse_args()
    preprocess(args)
