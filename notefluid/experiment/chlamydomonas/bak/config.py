import os


class Config:
    def __init__(self, split=True, progress=True, valid=True):
        self.split = split
        self.progress = progress
        self.valid = valid


class DetectConfig:
    def __init__(self, video_path, output_dir, debug=False):
        self.debug = debug
        self.video_path = video_path
        self.video_name = os.path.basename(self.video_path).split('.')[0]
        self.output_dir = f'{output_dir}/{self.video_name}'

        self.image_origin = f'{self.output_dir}/origin'
        self.image_detect = f'{self.output_dir}/detect'

        self.contain_origin = f'{self.output_dir}/contain_origin.pkl'
        self.recognize_path = f'{self.output_dir}/recognize.pkl'
        self.result_list = []

        self.mkdir(self.output_dir)
        self.mkdir(self.image_origin)
        self.mkdir(self.image_detect)

    def mkdir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
