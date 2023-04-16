import os

import pandas as pd


class BaseCache:
    def __init__(self, filepath="", overwrite=False):
        self.filepath = filepath
        self.overwrite = overwrite

    def exists(self):
        return os.path.exists(self.filepath)

    def execute(self):
        raise Exception("not implement.")

    def read(self):
        raise Exception("not implement.")

    def write(self):
        raise Exception("not implement.")


class CSVCache(BaseCache):
    def __init__(self, *args, **kwargs):
        super(CSVCache, self).__init__(*args, **kwargs)
        self.df = None

    def read(self):
        if self.overwrite or not self.exists():
            self.df = self.execute()
        if self.df is None:
            self.df = pd.read_csv(self.filepath)
        return self.df

    def write(self):
        self.df.to_csv(self.filepath, index=None)
