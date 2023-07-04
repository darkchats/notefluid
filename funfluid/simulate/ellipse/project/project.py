import math
import os

import pandas as pd


class BaseProject:
    def __init__(self, path):
        self.path = path

    @staticmethod
    def _load(path, index=0):
        df = pd.read_csv(path, sep='\s+', header=None)
        cols = [f"c{i}" for i in df.columns]
        cols[0] = 'x'
        cols[1] = 'y'
        cols[4] = 'theta'
        df.columns = cols
        df['theta'] = (df['theta']) * math.pi / 180.
        df = df.reset_index(names='step')
        df['index'] = index
        return df

    @property
    def orientation_files(self):
        results = []
        for file in os.listdir(self.path):
            if file.startswith("orientation"):
                results.append(os.path.join(self.path, file))
        results.sort(key=lambda x: x)
        return results

    def output_path(self, sub_path=''):
        path = os.path.join(self.path, "output", sub_path)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @property
    def project_name(self):
        return os.path.basename(self.path)
