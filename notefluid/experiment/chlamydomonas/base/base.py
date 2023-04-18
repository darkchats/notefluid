import json
import logging
import os
import pickle

import cv2
from tqdm import tqdm

from notefluid.common.base.cache import BaseCache


class VideoBase(BaseCache):
    def __init__(self, video_path, cache_dir='./cache_dir',
                 start_second=0, end_second=5 * 3600 * 1000, *args, **kwargs):
        self.video_path = video_path
        self.cache_dir = f"{cache_dir}/{os.path.basename(video_path)}"
        super(VideoBase, self).__init__(filepath=f"{self.cache_dir}/base.pkl", *args, **kwargs)

        self.start_second = start_second
        self.end_second = end_second

        self.video_width = 0
        self.video_height = 0
        self.frame_count = 0
        self.load_config()

    def load_config(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        config_path = self.video_path.replace(".avi", ".json")
        if not os.path.exists(config_path):
            return
        data = json.loads(open(config_path, 'r').read())
        if 'startSecond' in data.keys():
            self.start_second = data['startSecond']
        if 'endSecond' in data.keys():
            self.end_second = data['endSecond']

    def execute(self, *args, **kwargs):
        camera = cv2.VideoCapture(self.video_path)
        self.video_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rate = camera.get(cv2.CAP_PROP_FPS)
        frame_counter = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))

        pbar = tqdm(range(int(frame_counter / rate * 1000)))
        step = 0
        while True:
            step += 1
            res, image = camera.read()
            if not res:
                break
            millisecond = int(camera.get(cv2.CAP_OPENNI_DEPTH_MAP))
            if millisecond > 0:
                pbar.update(millisecond - pbar.n)
        self.frame_count = step
        camera.release()
        pbar.close()

    def print(self):
        logging.info(f"config-second:{self.start_second}->{self.end_second}")

    def _read(self, *args, **kwargs):
        with open(self.filepath, 'rb') as fr:
            self.video_width = pickle.load(fr)
            self.video_height = pickle.load(fr)
            self.frame_count = pickle.load(fr)

    def _save(self, *args, **kwargs):
        with open(self.filepath, 'wb') as fw:
            pickle.dump(self.video_width, fw)
            pickle.dump(self.video_height, fw)
            pickle.dump(self.frame_count, fw)


def process_wrap(fun, config: VideoBase, desc='process'):
    camera = cv2.VideoCapture(config.video_path)
    rate = camera.get(cv2.CAP_PROP_FPS)
    frame_counter = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
    desc = f'{os.path.basename(config.video_path)}-{desc}'
    pbar = tqdm(range(int(frame_counter / rate * 1000)))
    step = 0
    while True:
        step += 1
        res, image = camera.read()
        if not res:
            break
        image = cv2.GaussianBlur(image, (3, 3), 1)
        millisecond = int(camera.get(cv2.CAP_OPENNI_DEPTH_MAP))
        if millisecond > 0:
            pbar.update(millisecond - pbar.n)
        if millisecond < config.start_second * 1000:
            continue
        elif millisecond > config.end_second * 1000:
            break
        ext_json = {
            'millisecond': millisecond
        }
        msg = fun(step, image, ext_json)
        pbar.set_description(f'{desc}-{msg}')

    camera.release()
    pbar.close()


class BaseProgress:
    def __init__(self, config: VideoBase):
        self.config = config
        self.config.load_config()

    def process(self, overwrite=False, debug=False, *args, **kwargs):
        self.config.read(overwrite=overwrite)
        self.save(overwrite=overwrite)

    def process_background_video(self, overwrite=False, debug=False, *args, **kwargs):
        pass

    def process_contain_video(self, overwrite=False, debug=False, *args, **kwargs):
        pass

    def process_particle_video(self, overwrite=False, debug=False, *args, **kwargs):
        pass

    def save(self, overwrite=False, *args, **kwargs):
        return self.config.save(overwrite=overwrite)

    def load(self, overwrite=False, *args, **kwargs):
        return self.config.read(overwrite=overwrite)
