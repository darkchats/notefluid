import logging
import os
import pickle
from logging import root

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from notefluid.experiment.chlamydomonas.bak.fit import FitCircle, FitEllipse, FitBase


class Config:
    def __init__(self, split=True, progress=True, valid=True):
        self.split = split
        self.progress = progress
        self.valid = valid


class VideoProcess:
    def __init__(self, video_path, output_dir, config: Config = None, debug=False):
        self.debug = debug
        self.video_path = video_path
        self.video_name = os.path.basename(self.video_path).split('.')[0]
        self.output_dir = f'{output_dir}/{self.video_name}'
        self.recognize_path = f'{self.output_dir}/recognize.pkl'
        self.config = config or Config()
        self.result_list = []

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            os.makedirs(f'{self.output_dir}/image')
            os.makedirs(f'{self.output_dir}/image2')

    def image_split(self):
        output_dir = f'{self.output_dir}/image'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        camera = cv2.VideoCapture(self.video_path)
        for step in tqdm(range(int(camera.get(cv2.CAP_PROP_FRAME_COUNT)))):
            res, image = camera.read()
            if not res:
                break
            cv2.imwrite(f"{output_dir}/frame-{str(step + 1)}.png", image)
        camera.release()

    def process_video(self):
        camera = cv2.VideoCapture(self.video_path)
        fps = int(round(camera.get(cv2.CAP_PROP_FPS)))
        frame_counter = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))

        track_pre = None
        for step in tqdm(range(frame_counter)):
            res, image = camera.read()
            if not res:
                break
            data = {
                "video_path": self.video_path,
                "frame": step + 1,
                "time": (step + 1) / fps
            }

            data = self.process_pic(image, data_json=data, track_pre=track_pre)
            if 'ellipse' in data.keys():
                track_pre = data
            self.result_list.append(data)
        camera.release()

        with open(self.recognize_path, 'wb') as fw:
            pickle.dump(self.result_list, fw)

    def process_pic(self, img, data_json=None, track_pre=None, debug=False):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转为灰度值图
        # THRESH_OTSU, THRESH_TRIANGLE
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_TRIANGLE)  # 转为二值图
        # ret, binary = cv2.threshold(gray, img.max() * 0.92, 255, cv2.THRESH_BINARY)  # 转为二值图
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
        if debug:
            cv2.imshow("binary", binary)

        kernel = np.ones((2, 2), np.int8)
        binary = cv2.dilate(binary, kernel, iterations=1)
        if debug:
            cv2.imshow("dilate", binary)

        result = data_json or {}

        circle_pre, ellipse_pre = None, None
        if track_pre is not None:
            if 'ellipse' in track_pre.keys():
                ellipse_pre = FitEllipse(**track_pre['ellipse'])

        for i, contour in enumerate(contours):
            fit_base = FitBase(contour).fit()
            if not fit_base.valid():
                continue

            fit_circle = FitCircle(**fit_base.to_json()).fit()
            logging.debug(f'circle\t{fit_circle}')

            if fit_circle.valid():
                result['circle'] = fit_circle.to_json()
                continue

            fit_ellipse = FitEllipse(**fit_base.to_json()).fit()
            logging.debug(f'ellipse\t{fit_ellipse}')
            if debug:
                temp = np.zeros(img.shape, np.uint8)  # 生成黑背景
                fit_ellipse.draw(temp)
                cv2.imshow(f'ellipse-{i}', temp)
            if fit_ellipse.valid(ellipse_pre):
                if 'ellipse' not in result.keys():
                    result['ellipse'] = fit_ellipse.to_json()
                elif 'ellipse' in result.keys() and fit_ellipse.fit_score[1] > result['ellipse']['score'][1]:
                    result['ellipse'] = fit_ellipse.to_json()
                continue
        if debug:
            if 'circle' in result.keys():
                FitCircle(**result['circle']).print()
                FitCircle(**result['circle']).draw(img)
            if 'ellipse' in result.keys():
                FitEllipse(**result['ellipse']).print()
                FitEllipse(**result['ellipse']).draw(img)
            cv2.imshow("img", img)
        logging.info(f"{result.get('frame', 0)}, {result.keys()}")
        return result

    def get_frame(self, frame):
        path = f"{self.output_dir}/image/frame-{frame}.png"
        image = None
        if os.path.exists(path):
            image = cv2.imread(path)
        else:
            camera = cv2.VideoCapture(self.video_path)
            for step in tqdm(range(int(camera.get(cv2.CAP_PROP_FRAME_COUNT)))):
                res, image = camera.read()
                if not res:
                    break
                if step + 1 == frame:
                    break
            camera.release()
            cv2.imwrite(path, image)
        return image

    def pic_show(self, data_json, debug=False):
        image = self.get_frame(data_json['frame'])
        if image is None:
            return
        # image = cv2.imread(data_json['path'])  # 读取图像
        # temp = np.zeros(img.shape, np.uint8)  # 生成黑背景
        if 'circle' in data_json.keys():
            circle = FitCircle(**data_json['circle'])
            if debug:
                circle.print()
            circle.draw(image)
        if 'ellipse' in data_json.keys():
            ellipse = FitEllipse(**data_json['ellipse'])
            if debug:
                ellipse.print()
            ellipse.draw(image)
        cv2.imwrite(f"{self.output_dir}/image2/frame-{data_json['frame']}.png", image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if debug:
            cv2.imshow(f"image", image)

    def valid(self, debug=False):
        with open(self.recognize_path, 'rb') as fr:
            datas = pickle.load(fr)

            for data in tqdm(datas):
                if debug:
                    print(data['frame'], data.keys())
                # self.pic_show(data)
            # self.pic_show(datas[31])

    def load(self):
        with open(self.recognize_path, 'rb') as fr:
            self.result_list = pickle.load(fr)

    @property
    def ellipse_path(self):
        data = []
        for item in self.result_list:
            if 'ellipse' in item.keys():
                data.append([item['frame'], *item['ellipse']['center']])

        df = pd.DataFrame(data)
        df.columns = ['frame', 'x', 'y']
        df.to_csv(f'{self.output_dir}/ellipse_path.csv', index=None)
        return df

    def run(self):
        if self.config.split:
            self.image_split()
        if self.config.progress:
            self.process_video()
        else:
            self.load()
        if self.config.valid:
            self.valid(True)


root.setLevel(logging.WARN)
logging.basicConfig()

path_root = '/Users/chen/data/experiment/'
path = f"{path_root}/11-5-01003.avi"

dir_name = f'{path_root}/output'

config = Config(split=False, progress=False, valid=True)

video = VideoProcess(path, dir_name, config)
video.run()
video.load()
root.setLevel(logging.DEBUG)
print(video.ellipse_path)
video.process_pic(video.get_frame(1356), debug=True)
cv2.waitKey()
cv2.destroyAllWindows()
