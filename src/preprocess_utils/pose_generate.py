# use https://www.kaggle.com/datasets/burnoutminer/heights-and-weights-dataset
import cv2
import numpy as np
import mediapipe as mp


class PoseGenerate:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

    def __del__(self):
        self.pose.close()

    def generate(self, image_path: str, dist_path: str):
        image = cv2.imread(image_path)
        # OpenCVとMediaPipeでRGBの並びが違うため、処理前に変換しておく。
        # CV2:BGR → MediaPipe:RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        assert results.pose_landmarks, "No pose landmarks found"
        if results.pose_landmarks:
            # 黒背景の画像を作成
            height, width, _ = image.shape
            black_image = np.zeros((height, width, 3), dtype=np.uint8)

            # 全身の主要ランドマークを取得
            keypoints = [
                self.mp_pose.PoseLandmark.NOSE,
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.LEFT_ELBOW,
                self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                self.mp_pose.PoseLandmark.LEFT_WRIST,
                self.mp_pose.PoseLandmark.RIGHT_WRIST,
                self.mp_pose.PoseLandmark.LEFT_HIP,
                self.mp_pose.PoseLandmark.RIGHT_HIP,
                self.mp_pose.PoseLandmark.LEFT_KNEE,
                self.mp_pose.PoseLandmark.RIGHT_KNEE,
                self.mp_pose.PoseLandmark.LEFT_ANKLE,
                self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            ]
            # 座標を格納するリスト作成
            points = []
            # ランドマークを黒背景の画像にプロット
            for point in keypoints:
                landmark = results.pose_landmarks.landmark[point]
                cx = int(landmark.x * width)
                cy = int(landmark.y * height)
                points.append((cx, cy))
                cv2.circle(black_image, (cx, cy), 5, (0, 255, 0), -1)
            # 関節を結ぶ線を引く
            pairs = [
                (
                    self.mp_pose.PoseLandmark.NOSE,
                    self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                ),
                (
                    self.mp_pose.PoseLandmark.NOSE,
                    self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                ),
                (
                    self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                    self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                ),
                (
                    self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                    self.mp_pose.PoseLandmark.LEFT_ELBOW,
                ),
                (
                    self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                ),
                (
                    self.mp_pose.PoseLandmark.LEFT_ELBOW,
                    self.mp_pose.PoseLandmark.LEFT_WRIST,
                ),
                (
                    self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                    self.mp_pose.PoseLandmark.RIGHT_WRIST,
                ),
                (
                    self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                    self.mp_pose.PoseLandmark.LEFT_HIP,
                ),
                (
                    self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    self.mp_pose.PoseLandmark.RIGHT_HIP,
                ),
                (
                    self.mp_pose.PoseLandmark.LEFT_HIP,
                    self.mp_pose.PoseLandmark.RIGHT_HIP,
                ),
                (
                    self.mp_pose.PoseLandmark.LEFT_HIP,
                    self.mp_pose.PoseLandmark.LEFT_KNEE,
                ),
                (
                    self.mp_pose.PoseLandmark.RIGHT_HIP,
                    self.mp_pose.PoseLandmark.RIGHT_KNEE,
                ),
                (
                    self.mp_pose.PoseLandmark.LEFT_KNEE,
                    self.mp_pose.PoseLandmark.LEFT_ANKLE,
                ),
                (
                    self.mp_pose.PoseLandmark.RIGHT_KNEE,
                    self.mp_pose.PoseLandmark.RIGHT_ANKLE,
                ),
            ]

            # ランドマークのインデックスを含んだペアを使って線を引く
            for pair in pairs:
                start_point = results.pose_landmarks.landmark[pair[0]]
                end_point = results.pose_landmarks.landmark[pair[1]]

                start = (int(start_point.x * width), int(start_point.y * height))
                end = (int(end_point.x * width), int(end_point.y * height))

                cv2.line(black_image, start, end, (0, 255, 0), 2)

            cv2.imwrite(dist_path, black_image)
