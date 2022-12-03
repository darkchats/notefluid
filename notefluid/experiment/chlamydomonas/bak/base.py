import os

import cv2
from tqdm import tqdm

from notefluid.experiment.chlamydomonas.bak.config import DetectConfig


class BaseDetect:
    def __init__(self, conf: DetectConfig):
        self.conf = conf

    def get_frame(self, frame):
        path = f"{self.conf.image_origin}/frame-{frame}.png"
        image = None
        if os.path.exists(path):
            image = cv2.imread(path)
        else:
            camera = cv2.VideoCapture(self.conf.video_path)
            frame_counter = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))

            for step in tqdm(range(frame_counter)):
                res, image = camera.read()
                if not res:
                    break
                if step + 1 == frame:
                    break
            camera.release()
            cv2.imwrite(path, image)
        return image
