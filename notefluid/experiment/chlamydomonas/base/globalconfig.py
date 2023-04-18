import os


class GlobalConfig:
    def __init__(self, path_root):
        # 总得路径
        self.path_root = path_root
        # 视频路径
        self.videos_dir = f'{self.path_root}/videos'
        # 结果保持位置
        self.results_dir = f'{self.path_root}/results'
        # 总的结果文件
        self.results_json = f'{self.path_root}/result.json'

    def get_result_path(self, video_path):
        return os.path.dirname(video_path.replace(self.videos_dir, self.results_dir))
