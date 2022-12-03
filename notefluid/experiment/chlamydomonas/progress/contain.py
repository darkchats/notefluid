import os.path
import pickle
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

from notefluid.experiment.chlamydomonas.progress.base import BaseProgress
from notefluid.utils.log import logger


def fit_contain(contour):
    # length = round(cv2.arcLength(contour, True), 2)  # 获取轮廓长度

    if len(contour) < 5000:
        logger.debug(f"size: {len(contour)}<10")
        return None, None
    area = round(cv2.contourArea(contour), 2)  # 获取轮廓面积
    if area < 100000:
        logger.debug(f"area: {area} < 100000")
        return None, None
    elif area > 1000000:
        logger.debug(f"area: {area} > 1000000")
        return None, None

    # (x,y) 代表椭圆中心点的位置, radius 代表半径
    center, radius = cv2.minEnclosingCircle(contour)
    data = np.subtract(np.reshape(contour, [contour.shape[0], 2]), np.array([center]))
    score1 = 1 - round(np.abs((np.linalg.norm(data, axis=1) - radius).mean()) / radius, 4)
    if score1 < 0.8:
        return None, None
    score2 = 1 - abs(radius * radius * np.pi / area - 1)
    if score2 < 0.8:
        return None, None
    return np.array(center), radius


class BackContain:
    def __init__(self):
        self.center = np.array([0, 0])
        self.radius = 0
        self.count = 0

    def add(self, center, radius):
        if self.count == 0:
            self.count += 1
            self.center = center
            self.radius = radius
            return

        self.center = self.center * self.count + center
        self.radius = self.radius * self.count + radius
        self.count += 1
        self.center /= self.count
        self.radius /= self.count
        return self

    def valid(self, center, radius):
        s1 = np.linalg.norm(self.center - center)
        if s1 > 10:
            return False
        s2 = abs(self.radius - radius)
        if s2 > 10:
            return False
        return True


class BackContainList(BaseProgress):
    def __init__(self, *args, **kwargs):
        super(BackContainList, self).__init__(*args, **kwargs)
        self.backcontain_path = f'{self.cache_dir}/backcontain.pkl'
        self.backcontain_list: List[BackContain] = []

    def add_image(self, image) -> List[BackContain]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度值图
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # 转为二值图
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
        contains = []
        for i, contour in enumerate(contours):
            center, radius = fit_contain(contour)
            if center is not None:
                for contain in self.backcontain_list[::-1]:
                    if contain.valid(center, radius):
                        contain.add(center, radius)
                        contains.append(contain)
                        break
                contain = BackContain()
                contain.add(center, radius)
                self.backcontain_list.append(contain)
                contains.append(contain)
                logger.info("add a newer")
        return contains

    def process(self, overwrite=False, *args, **kwargs):
        super(BackContainList, self).process(overwrite=overwrite, *args, **kwargs)
        if self.load(overwrite=overwrite, *args, **kwargs):
            return
        camera = cv2.VideoCapture(self.video_path)
        frame_counter = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(range(frame_counter))

        filename = os.path.basename(self.video_path)
        for step in pbar:
            res, image = camera.read()
            if not res:
                break
            self.add_image(image)
            pbar.set_description(f"backcontain-{filename}-{len(self.backcontain_list)}")
        camera.release()

    def save(self, overwrite=False, *args, **kwargs):
        super(BackContainList, self).save(overwrite=overwrite, *args, **kwargs)
        if not overwrite and os.path.exists(self.backcontain_path):
            return False
        with open(self.backcontain_path, 'wb') as fw:
            pickle.dump(self.backcontain_list, fw)

    def load(self, overwrite=False, *args, **kwargs):
        super(BackContainList, self).load(overwrite=overwrite, *args, **kwargs)
        if overwrite or not os.path.exists(self.backcontain_path):
            return False
        with open(self.backcontain_path, 'rb') as fr:
            self.backcontain_list = pickle.load(fr)
        return True
