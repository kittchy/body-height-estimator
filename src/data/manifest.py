from pydantic import BaseModel


class Manifest(BaseModel):
    original_image_path: str
    depth_image_path: str
    mask_image_path: str
    pose_image_path: str
    height: float

    @staticmethod
    def load_jsonp():
        pass
