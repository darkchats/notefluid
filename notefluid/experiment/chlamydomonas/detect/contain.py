import json
import math
import os.path
from typing import List

import cv2
import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from notefluid.common.base.cache import CSVDataFrameCache
from notefluid.experiment.chlamydomonas.base.base import VideoBase
from notefluid.experiment.chlamydomonas.detect.background import BackGroundDetect, BackGround
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
    def __init__(self, center=None, radius=0, count=0, uid=0):
        self.center = center or np.array([0, 0])
        self.radius = radius
        self.count = count
        self.uid = uid

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

    def is_inside(self, center) -> bool:
        dis = self.cul_distance(center)
        return dis < self.radius

    def cul_distance(self, center):
        return math.sqrt((center[0] - self.center[0]) ** 2 + (center[1] - self.center[1]) ** 2)

    def to_json(self):
        return {
            "centerX": self.center[0],
            "centerY": self.center[1],
            "radius": self.radius,
            "count": self.count,
            "uid": self.uid
        }

    def parse(self, data):
        self.center[0] = data['centerX']
        self.center[1] = data['centerY']
        self.radius = data['radius']
        self.count = data['count']
        self.uid = data['uid']
        return self


class ContainDetect(CSVDataFrameCache):
    def __init__(self, config: VideoBase, *args, **kwargs):
        self.config = config
        super(ContainDetect, self).__init__(filepath=f'{self.config.cache_dir}/detect_contains.csv', *args, **kwargs)
        self.contain_list: List[BackContain] = []

    def process_contain_image(self, background: BackGround, debug=False) -> List[BackContain]:
        gray = cv2.cvtColor(background.back_image, cv2.COLOR_BGR2GRAY)  # 转为灰度值图
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_TRIANGLE)  # 转为二值图
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓

        result_contain = None
        for i, contour in enumerate(contours):
            center, radius = fit_contain(contour)
            if center is not None:
                if debug:
                    cv2.circle(background.back_image, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 2)
                    cv2.circle(binary, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 2)

                contain = BackContain(uid=background.uid)
                contain.add(center, radius)
                if result_contain is None or contain.radius < result_contain.radius:
                    result_contain = contain

        if debug and result_contain is not None:
            cv2.imshow(f"{os.path.basename(self.config.video_path)}-binary", binary)
            cv2.imshow(f"{os.path.basename(self.config.video_path)}-image", background.back_image)
            cv2.waitKey()
        return [result_contain] if result_contain is not None else []

    def _execute(self, backgrounds: BackGroundDetect, overwrite=False, debug=False, *args, **kwargs):
        for step, background in tqdm(enumerate(backgrounds.background_list), desc='detect contain'):
            contains = self.process_contain_image(background, debug=debug)
            self.contain_list.extend(contains)
        self.df = pd.DataFrame([par.to_json() for par in self.contain_list])

    def find_contain(self, uid) -> BackContain:
        for contain in self.contain_list:
            if contain.uid == uid:
                return contain
        raise Exception(f'cannot find contain {uid}')

    def _parse(self, df: DataFrame, *args, **kwargs):
        for record in json.loads(df.to_json(orient='records')):
            self.df = df
            self.contain_list.append(BackContain().parse(record))
