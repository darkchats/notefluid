import os.path
import pickle
from typing import List

import cv2
import numpy as np
import pandas as pd

from notefluid.common.base.cache import BaseCache
from notefluid.experiment.chlamydomonas.base.base import process_wrap, VideoBase
from notefluid.experiment.chlamydomonas.detect.background import BackGroundDetect
from notefluid.experiment.chlamydomonas.detect.contain import ContainDetect
from notefluid.utils.log import logger


def fit_particle(contour):
    if len(contour) < 50:
        logger.debug(f"size: {len(contour)}<50")
        return None, None, None
    area = round(cv2.contourArea(contour), 2)  # 获取轮廓面积
    if area < 100:
        logger.debug(f"area: {area} < 100")
        return None, None, None

    if len(contour) < 10:
        logger.debug(f"size: {len(contour)}<10")
        return None, None, None
    elif area > 100000:
        return None, None, None

    # (x,y) 代表椭圆中心点的位置, radius 代表半径
    center, radius, angle = cv2.fitEllipse(contour)
    s1 = 1 - abs(radius[0] * radius[1] * np.pi / 4 / area - 1)
    if s1 < 0.3:
        return None, None, None
    if radius[0] / radius[1] > 2 or radius[0] / radius[1] < 0.5:
        return None, None, None
    return center, radius, angle


class Particle:
    def __init__(self, contour, center, radius, angle, step, ext_json):
        self.contour = contour
        self.center = center
        self.radius = radius
        self.angle = angle
        self.step = step
        self.millisecond = ext_json.get("millisecond", 0)
        self.background_uid = ext_json.get("background_uid", 0)

    def to_json(self):
        return {
            "step": self.step,
            "centerX": self.center[0],
            "centerY": self.center[1],
            "radiusA": self.radius[0],
            "radiusB": self.radius[1],
            "angle": self.angle,
            "millisecond": self.millisecond,
            "background_uid": self.background_uid
        }


class ParticleDetect(BaseCache):
    def __init__(self, config: VideoBase, *args, **kwargs):
        self.config = config
        super(ParticleDetect, self).__init__(
            filepath=f'{self.config.cache_dir}/detect_particles.pkl', *args, **kwargs)
        self.particle_csv_path = f'{self.config.cache_dir}/detect_particles.csv'
        self.particle_list: List[Particle] = []

    def process_particle_image(self,
                               backgrounds: BackGroundDetect,
                               contains: ContainDetect,
                               image, step, ext_json) -> List[Particle]:
        background = backgrounds.process_background_nearest(image)
        ext_json['background_uid'] = background.uid

        def find_contain(uid):
            for contain in contains.contain_list:
                if contain.uid == uid:
                    return contain.radius
                raise Exception(f'{self.filename2} cannot find contain {uid}')

        image = np.abs(background.back_image.astype(np.int) - image.astype(np.int))
        image = image.astype(np.uint8)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度值图
        # THRESH_OTSU, THRESH_TRIANGLE
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_TRIANGLE)  # 转为二值图
        # ret, binary = cv2.threshold(gray, img.max() * 0.92, 255, cv2.THRESH_BINARY)  # 转为二值图
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓

        particles = []
        contain_rate = 150.0 / find_contain(background.uid)
        for i, contour in enumerate(contours):
            center, radius, angle = fit_particle(contour)

            if center is not None:
                center = (center[0] * contain_rate, center[1] * contain_rate)
                radius = (radius[0] * contain_rate, radius[1] * contain_rate)

                particle = Particle(contour=contour,
                                    center=center,
                                    radius=radius,
                                    angle=angle,
                                    step=step,
                                    ext_json=ext_json)
                self.particle_list.append(particle)
                particles.append(particle)
        return particles

    def _execute(self,
                 backgrounds: BackGroundDetect,
                 contains: ContainDetect,
                 debug=False, *args, **kwargs):
        def fun(step, image, ext_json):
            particles = self.process_particle_image(backgrounds, contains, image, step, ext_json)
            if debug:
                for particle in particles:
                    cv2.ellipse(image, (particle.center, particle.radius, particle.angle), (0, 255, 0), 2)
                cv2.imshow(f'{os.path.basename(self.config.video_path)}', image)
                cv2.waitKey(delay=10)
            return len(self.particle_list)

        process_wrap(fun, self.config, desc='detect particle')

    @property
    def particle_df(self):
        return pd.DataFrame([par.to_json() for par in self.particle_list])

    def _save(self, *args, **kwargs):
        with open(self.filepath, 'wb') as fw:
            pickle.dump(self.particle_list, fw)
        self.particle_df.to_csv(self.particle_csv_path, index=False)

    def _read(self, *args, **kwargs):
        with open(self.filepath, 'rb') as fr:
            self.particle_list = pickle.load(fr)
        return True
