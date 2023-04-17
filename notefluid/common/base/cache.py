import os
import pickle

import pandas as pd

from notefluid.utils.log import logger


class BaseCache:
    def __init__(self, filepath="", *args, **kwargs):
        self.filepath = filepath

    @property
    def filename(self):
        return os.path.basename(self.filepath)

    def exists(self):
        return os.path.exists(self.filepath)

    def _execute(self, *args, **kwargs):
        raise Exception("not implement.")

    def execute(self, *args, **kwargs):
        return self._execute(*args, **kwargs)

    def _read(self, *args, **kwargs):
        raise Exception("not implement.")

    def _write(self, *args, **kwargs):
        raise Exception("not implement.")

    def read(self, overwrite=False, *args, **kwargs):
        if overwrite:
            logger.info("overwrite,execute...")
            self.execute(*args, **kwargs)
            self.write(overwrite=overwrite, *args, **kwargs)
        elif not self.exists():
            logger.info(f"{self.filename} not exists,execute...")
            self.execute(*args, **kwargs)
            self.write(overwrite=overwrite, *args, **kwargs)
        return self._read(*args, **kwargs)

    def write(self, overwrite=False, *args, **kwargs):
        if not self.exists():
            logger.info(f"{self.filename}  file not exists,save.")
            self._write(*args, **kwargs)
        elif overwrite:
            logger.info("file exists,overwrite.")
            self._write(*args, **kwargs)


class BaseDataFrameCache(BaseCache):
    def __init__(self, *args, **kwargs):
        super(BaseDataFrameCache, self).__init__(*args, **kwargs)
        self.df = None

    def execute(self, *args, **kwargs):
        self.df = self._execute(*args, **kwargs)
        return self.df

    def read(self, overwrite=False, *args, **kwargs):
        if overwrite:
            logger.info("overwrite,execute...")
            self.df = self.execute(*args, **kwargs)
            self.write(overwrite=overwrite, *args, **kwargs)
        elif not self.exists():
            logger.info(f"{self.filename} not exists,execute...")
            self.df = self.execute(*args, **kwargs)
            self.write(overwrite=overwrite, *args, **kwargs)
        elif self.df is None:
            self.df = self._read(*args, **kwargs)
        return self.df


class CSVDataFrameCache(BaseDataFrameCache):
    def __init__(self, *args, **kwargs):
        super(CSVDataFrameCache, self).__init__(*args, **kwargs)
        self.df = None

    def _read(self, *args, **kwargs):
        self.df = pd.read_csv(self.filepath)
        return self.df

    def _write(self, *args, **kwargs):
        if self.df is None:
            logger.info("df is None.")
            return
        self.df.to_csv(self.filepath, index=None)


class PickleDataFrameCache(BaseDataFrameCache):
    def __init__(self, *args, **kwargs):
        super(PickleDataFrameCache, self).__init__(*args, **kwargs)
        self.df = None

    def _read(self, *args, **kwargs):
        with open(self.filepath, 'rb') as fr:
            self.df = pickle.load(fr)
        return self.df

    def _write(self, *args, **kwargs):
        with open(self.filepath, 'wb') as fw:
            pickle.dump(self.df, fw)
