from pydantic import BaseModel
import json


class Manifest(BaseModel):
    original_image_path: str
    depth_image_path: str
    mask_image_path: str
    pose_image_path: str
    height: float

    @staticmethod
    def load_jsonp(filepath: str) -> list["Manifest"]:
        manifests = []
        with open(filepath) as f:
            for line in f.readlines():
                data = json.loads(line)
                manifests.append(Manifest(**data))
        return manifests
