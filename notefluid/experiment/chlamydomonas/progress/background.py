import os
import pickle
from typing import List

import numpy as np

from notefluid.experiment.chlamydomonas.progress.base import BaseProgress
from notefluid.utils.log import logger


class BackGround:
    def __init__(self, length, width, height=3, uid=None):
        self.back_image = np.zeros([width, length, height])
        self.count = 0
        self.uid = uid

    def add(self, image):
        if self.count == 0:
            self.count += 1
            self.back_image = image
            return
            # self.back_image = self.back_image * self.count + image
        self.count += 1
        # self.back_image /= self.count
        self.back_image = np.max(np.array([self.back_image, image]), axis=0)
        return self

    def valid(self, image, s1=0.4, s2=0.03):
        if self.count == 0:
            return True

        res1 = np.abs(self.back_image.astype(np.int) - image.astype(np.int))
        res = abs(np.sum(res1 > (image.max() - image.min()) * s1))
        if 1.0 * res / image.shape[0] / image.shape[1] < s2:
            return True
        return False

    def score(self, image):
        res1 = np.abs(self.back_image.astype(np.int) - image.astype(np.int))
        return np.sum(res1 > 30)

    def __str__(self):
        return f"{self.uid}    {self.count}    {self.back_image.shape}"


class BackGroundList(BaseProgress):
    def __init__(self, *args, **kwargs):
        super(BackGroundList, self).__init__(*args, **kwargs)
        self.background_path = f'{self.cache_dir}/background.pkl'
        self.background_list: List[BackGround] = []

    def process_background_nearest(self, image, debug=False, *args, **kwargs) -> BackGround:
        back = None
        min_score = 9999999999990
        for back in self.background_list[::-1]:
            score = back.score(image)
            if score < min_score:
                min_score = score
                back = back
        return back

    def process_background_image(self, step, image, debug=False, *args, **kwargs) -> BackGround:
        for back in self.background_list[::-1]:
            if back.valid(image):
                back.add(image)
                return back
        back = BackGround(self.video_width, self.video_height, uid=step)
        back.add(image)
        self.background_list.append(back)
        logger.debug("add a newer")
        return back

    def process_background_video(self, overwrite=False, debug=False, *args, **kwargs):
        super(BackGroundList, self).process(overwrite=overwrite, *args, **kwargs)
        if self.load_background(overwrite=overwrite, *args, **kwargs):
            return

        def fun(step, image, ext_json):
            self.process_background_image(step, image)
            return len(self.background_list)

        self.process_wrap(fun, desc='background')
        self.save_background(overwrite=overwrite)

    def save_background(self, overwrite=False, *args, **kwargs):
        super(BackGroundList, self).save(overwrite=overwrite, *args, **kwargs)
        if not overwrite and os.path.exists(self.background_path) and len(self.background_list) > 0:
            return False
        with open(self.background_path, 'wb') as fw:
            pickle.dump(self.background_list, fw)

    def load_background(self, overwrite=False, *args, **kwargs):
        super(BackGroundList, self).load(overwrite=overwrite, *args, **kwargs)
        if overwrite or not os.path.exists(self.background_path):
            logger.info(f"overwrite or {self.background_path} not exists,return")
            return False
        with open(self.background_path, 'rb') as fr:
            self.background_list = pickle.load(fr)
            logger.info(f"load {self.background_path} success:{len(self.background_list)}")
        return True
