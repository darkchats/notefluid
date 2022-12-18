import os.path
import pickle
from typing import List

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from notefluid.experiment.chlamydomonas.progress.background import BackGroundList
from notefluid.utils.log import logger


def fit_contain(contour):
    # length = round(cv2.arcLength(contour, True), 2)  # 获取轮廓长度

    if len(contour) < 100:
        logger.debug(f"size: {len(contour)}<10")
        return None, None
    area = round(cv2.contourArea(contour), 2)  # 获取轮廓面积
    if area < 10000:
        logger.debug(f"area: {area} < 100000")
        return None, None
    elif area > 985000:
        logger.debug(f"area: {area} > 1000000")
        return None, None

    # (x,y) 代表椭圆中心点的位置, radius 代表半径
    center, radius = cv2.minEnclosingCircle(contour)
    # if radius > 600:
    #     return None, None
    data = np.subtract(np.reshape(contour, [contour.shape[0], 2]), np.array([center]))
    score1 = 1 - round(np.abs((np.linalg.norm(data, axis=1) - radius).mean()) / radius, 4)
    if score1 < 0.6:
        return None, None
    area2 = radius * radius * np.pi
    if area2 < area * 0.8:
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

    def to_json(self):
        return {
            "centerX": self.center[0],
            "centerY": self.center[1],
            "radius": self.radius,
        }


class BackContainList(BackGroundList):
    def __init__(self, *args, **kwargs):
        super(BackContainList, self).__init__(*args, **kwargs)
        self.backcontain_path = f'{self.cache_dir}/backcontain.pkl'
        self.backcontain_csv_path = f'{self.cache_dir}/backcontain.csv'
        self.backcontain_list: List[BackContain] = []

    def process_contain_image(self, image, step, ext_json, debug=False) -> List[BackContain]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度值图
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_TRIANGLE)  # 转为二值图
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓

        result_contain = None
        for i, contour in enumerate(contours):
            center, radius = fit_contain(contour)
            if center is not None:
                if debug:
                    cv2.circle(image, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 2)
                    cv2.circle(binary, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 2)

                contain = BackContain()
                contain.add(center, radius)
                if result_contain is None or contain.radius < result_contain.radius:
                    result_contain = contain

        if debug and result_contain is not None:
            cv2.imshow(f"{os.path.basename(self.video_path)}-binary", binary)
            cv2.imshow(f"{os.path.basename(self.video_path)}-image", image)
            cv2.waitKey()
        return [result_contain] if result_contain is not None else []

    def process_contain_video(self, overwrite=False, debug=False, *args, **kwargs):
        super(BackContainList, self).process(overwrite=False, *args, **kwargs)
        if self.load_contain(overwrite=overwrite, *args, **kwargs):
            return

        pbar = tqdm(enumerate(self.background_list))

        for step, background in pbar:
            image = background.back_image
            contains = self.process_contain_image(image, 0, None, debug=debug)
            self.backcontain_list.extend(contains)
        self.save_contain(overwrite=overwrite)

    def save_contain(self, overwrite=False, *args, **kwargs):
        super(BackContainList, self).save(overwrite=overwrite, *args, **kwargs)
        if not overwrite and os.path.exists(self.backcontain_path):
            return False
        with open(self.backcontain_path, 'wb') as fw:
            pickle.dump(self.backcontain_list, fw)
        data = [par.to_json() for par in self.backcontain_list]
        df = pd.DataFrame(data)
        df.to_csv(self.backcontain_csv_path, index=None)

    def load_contain(self, overwrite=False, *args, **kwargs):
        super(BackContainList, self).load(overwrite=overwrite, *args, **kwargs)
        if overwrite or not os.path.exists(self.backcontain_path):
            return False
        with open(self.backcontain_path, 'rb') as fr:
            self.backcontain_list = pickle.load(fr)
        return True
