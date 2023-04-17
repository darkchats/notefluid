class Config:
    def __init__(self, path_root):
        self.path_root = path_root
        self.videos_dir = f'{self.path_root}/videos'
        self.results_dir = f'{self.path_root}/results'
        self.results_json = f'{self.path_root}/result.json'
