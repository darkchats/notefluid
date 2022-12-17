import json
import logging
import os
import pickle

import cv2
from tqdm import tqdm


class VideoConfig:
    def __init__(self, start_second=0, end_second=5 * 3600 * 1000, *args, **kwargs):
        self.start_second = start_second
        self.end_second = end_second

    def load_config(self, config_path=None):
        if config_path is None:
            return
        if not os.path.exists(config_path):
            return
        data = json.loads(open(config_path, 'r').read())
        if 'startSecond' in data.keys():
            self.start_second = data['startSecond']
        if 'endSecond' in data.keys():
            self.end_second = data['endSecond']

    def print(self):
        logging.info(f"config-startSecond:{self.start_second}")
        logging.info(f"config-startSecond:{self.end_second}")


class BaseProgress:
    def __init__(self, video_path, config=None, cache_dir='./cache_dir'):
        self.cache_dir = f"{cache_dir}/{os.path.basename(video_path)}"
        self.video_path = video_path

        self.config: VideoConfig = VideoConfig()
        if isinstance(config, VideoConfig):
            self.config = config
        elif isinstance(config, str):
            self.config.load_config(config)
        else:
            self.config.load_config(self.video_path.replace(".avi", ".json"))
            self.config.print()

        self.video_width = 0
        self.video_height = 0
        self.basepath = f"{self.cache_dir}/base.pkl"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def process_wrap(self, fun, desc='process'):
        camera = cv2.VideoCapture(self.video_path)
        rate = camera.get(cv2.CAP_PROP_FPS)
        frame_counter = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
        desc = f'{os.path.basename(self.video_path)}-{desc}'
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
            if millisecond < self.config.start_second * 1000:
                continue
            elif millisecond > self.config.end_second * 1000:
                break
            ext_json = {
                'millisecond': millisecond
            }
            msg = fun(step, image, ext_json)
            pbar.set_description(f'{desc}-{msg}')

        camera.release()
        pbar.close()

    def process(self, overwrite=False, debug=False, *args, **kwargs):
        if self.load(overwrite=overwrite, *args, **kwargs):
            return
        camera = cv2.VideoCapture(self.video_path)
        self.video_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        camera.release()

    def process_background_video(self, overwrite=False, debug=False, *args, **kwargs):
        pass

    def process_contain_video(self, overwrite=False, debug=False, *args, **kwargs):
        pass

    def process_particle_video(self, overwrite=False, debug=False, *args, **kwargs):
        pass

    def save(self, overwrite=False, *args, **kwargs):
        if not overwrite and os.path.exists(self.basepath):
            return False
        with open(self.basepath, 'wb') as fw:
            pickle.dump(self.video_width, fw)
            pickle.dump(self.video_height, fw)
        return True

    def load(self, overwrite=False, *args, **kwargs):
        if overwrite or not os.path.exists(self.basepath):
            return False
        with open(self.basepath, 'rb') as fr:
            self.video_width = pickle.load(fr)
            self.video_height = pickle.load(fr)
        return True
