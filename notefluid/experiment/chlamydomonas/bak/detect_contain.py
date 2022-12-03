import math
import os.path
import pickle

import cv2
import numpy as np
from tqdm import tqdm

from notefluid.experiment.chlamydomonas.bak.base import BaseDetect
from notefluid.experiment.chlamydomonas.bak.domain import Contain, FitContain
from notefluid.utils.log import logger


class FindContain(BaseDetect):
    def __init__(self, *args, **kwargs):
        super(FindContain, self).__init__(*args, **kwargs)
        self.contain_set = []

    def detect_contain_video(self, overwrite=False, debug=False):
        logger.info(f"detect contain video start")
        contain_list = []
        if overwrite or not os.path.exists(self.conf.contain_origin):
            camera = cv2.VideoCapture(self.conf.video_path)
            frame_counter = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
            for _ in tqdm(range(frame_counter), desc='识别容器'):
                res, image = camera.read()
                if not res:
                    break
                contain_list.extend(self.detect_contain_pic(image, debug=debug))
            camera.release()
            with open(self.conf.contain_origin, 'wb') as fw:
                pickle.dump(contain_list, fw)
        else:
            with open(self.conf.contain_origin, 'rb') as fr:
                contain_list = pickle.load(fr)
        contain_set = []

        for i, coin in enumerate(contain_list):
            flag = True
            for contain in contain_set:
                if contain.valid(coin.center, coin.radius):
                    contain.merge(coin.center, coin.radius)
                    flag = False
                    continue
            if flag:
                contain_set.append(Contain(coin.center, coin.radius))
            logger.debug(f"{i}\t{len(contain_set)}\t{coin}")

        contain_set = sorted(contain_set, key=lambda x: x.count, reverse=True)
        for contain in contain_set:
            logger.info(contain)
        contain_set = contain_set[:2]
        contain_set = sorted(contain_set, key=lambda x: x.radius)
        contain_set = contain_set[:1]

        for contain in contain_set:
            logger.info(contain)

        self.contain_set = contain_set
        logger.info(f"detect contain video end")
        return contain_set

    def detect_contain_pic(self, img, debug=False):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转为灰度值图
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # 转为二值图
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
        if debug:
            show = np.hstack((gray, binary))
            cv2.imshow("show", show)

        contain_list = []

        for i, contour in enumerate(contours):
            fit_contain = FitContain(contour=contour).fit()
            if fit_contain.valid():
                contain_list.append(fit_contain)

                if debug:
                    temp = np.zeros(img.shape, np.uint8)
                    fit_contain.draw(temp)
                    cv2.imshow(f'ellipse-{i}', temp)

                    fit_contain.print()
                    fit_contain.draw(img)
                    cv2.imshow("img", img)
        return contain_list

    def detect_particle_video(self, overwrite=False, debug=False):
        logger.info(f"detect particle video start")
        camera = cv2.VideoCapture(self.conf.video_path)
        frame_counter = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
        height, width = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        background = np.zeros([height, width], dtype=np.uint8)

        y1, x1 = self.contain_set[0].center
        r1 = self.contain_set[0].radius
        for i in range(height):
            for j in range(width):
                dis = math.sqrt((x1 - i) ** 2 + (y1 - j) ** 2)
                if dis < r1 * 0.98:
                    background[i, j] = 1
        print(background.shape)
        for _ in tqdm(range(frame_counter)):
            res, image = camera.read()

            if not res:
                break

            self.detect_particle_pic(img=image, background=background, debug=debug)
            # time.sleep(1)
            cv2.waitKey(1000)

        camera.release()
        logger.info(f"detect particle video end")

    def detect_particle_pic(self, img, background=None, debug=False):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转为灰度值图
        gray_max = np.argmax(np.bincount(gray.reshape(-1)))

        gray = gray * background + (1 - background) * gray_max
        logger.debug(f'gray shape:{gray.shape}')
        logger.debug(f'background shape:{background.shape}')

        # THRESH_OTSU, THRESH_TRIANGLE
        # ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # 转为二值图
        # ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_TRIANGLE)  # 转为二值图
        ret, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  # 转为二值图

        contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
        if debug:
            show = np.hstack((gray, binary))
            cv2.imshow("show", show)

        contain_list = []
        if not os.path.exists(self.conf.contain_origin):
            for i, contour in enumerate(contours):

                fit_contain = FitContain(contour=contour).fit()
                if fit_contain.valid():
                    contain_list.append(fit_contain)

                    if debug:
                        temp = np.zeros(img.shape, np.uint8)
                        fit_contain.draw(temp)
                        cv2.imshow(f'ellipse-{i}', temp)

                        fit_contain.print()
                        fit_contain.draw(img)
                        cv2.imshow("img", img)
        return contain_list
