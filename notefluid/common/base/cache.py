import os

import pandas as pd


class BaseCache:
    def __init__(self, filepath="", overwrite=False):
        self.filepath = filepath
        self.overwrite = overwrite

    def exists(self):
        return os.path.exists(self.filepath)

    def execute(self, *args, **kwargs):
        raise Exception("not implement.")

    def read(self, *args, **kwargs):
        raise Exception("not implement.")

    def write(self, *args, **kwargs):
        raise Exception("not implement.")


class CSVCache(BaseCache):
    def __init__(self, *args, **kwargs):
        super(CSVCache, self).__init__(*args, **kwargs)
        self.df = None

    def read(self, *args, **kwargs):
        if self.overwrite or not self.exists():
            self.df = self.execute()
        if self.df is None:
            self.df = pd.read_csv(self.filepath)
        return self.df

    def write(self, *args, **kwargs):
        self.df.to_csv(self.filepath, index=None)
