import json
import math
import os

import pandas as pd
from tqdm import tqdm

from notefluid.common.base.cache import CSVCache
from notefluid.experiment.chlamydomonas.progress.particle import ParticleWithoutBackgroundList


class AnalyseParticle(ParticleWithoutBackgroundList):
    def __init__(self, *args, **kwargs):
        super(AnalyseParticle, self).__init__(*args, **kwargs)
        self.particle_track_path = f'{self.cache_dir}/particle_track.csv'
        self.particle_track = pd.DataFrame()

    def save_analyse(self, overwrite=False, *args, **kwargs):
        super(AnalyseParticle, self).save(overwrite=overwrite, *args, **kwargs)
        if not overwrite and os.path.exists(self.particle_track_path) and len(self.particle_list) > 0:
            return False
        self.particle_track.to_csv(self.particle_track_path, index=None)

    def load_analyse(self, overwrite=False, *args, **kwargs):
        super(ParticleWithoutBackgroundList, self).load(overwrite=overwrite, *args, **kwargs)
        if overwrite or not os.path.exists(self.particle_track_path):
            return False
        self.particle_track = pd.read_csv(self.particle_track_path)
        return True

    def analyse_track(self, overwrite=False, debug=False, *args, **kwargs):
        if len(self.backcontain_list) == 0 or not os.path.exists(self.particle_csv_path):
            return
        contains = pd.read_csv(self.backcontain_csv_path)
        particles = pd.read_csv(self.particle_csv_path)

        cx, cy, r = contains.values[0]

        def cul_dis(x, y, _x, _y):
            return math.sqrt((x - _x) * (x - _x) + (y - _y) * (y - _y))

        def cul_par_dis(row):
            x, y = row['centerX'], row['centerX']
            # cx, cy
            index = int(row['background_uid'] - 1)
            if index > len(contains):
                index = 0
            _cx, _cy, _cr = contains.values[index]
            return cul_dis(x, y, _cx, _cy)

        # 过滤容器外部的颗粒
        particles['dis'] = particles.apply(lambda x: cul_par_dis(x), axis=1)
        particles = particles[particles['dis'] <= r * 1.02]

        # 过滤跳跃式颗粒
        pre_line = None
        result = []
        for line in json.loads(particles.to_json(orient='records')):
            if pre_line is None:
                pre_line = line
                result.append(line)
                continue
            dis = cul_dis(line['centerX'], line['centerY'], pre_line['centerX'], pre_line['centerY'])
            if dis < 100:
                pre_line = line
                result.append(line)

        self.particle_track = pd.DataFrame(result)
        self.save_analyse(overwrite=overwrite)


class MSDCalculate(CSVCache):
    def __init__(self, *args, **kwargs):
        super(MSDCalculate, self).__init__(*args, **kwargs)

    def execute(self, df, *args, **kwargs):
        msd_result = []
        for i in tqdm(range(1, len(df))):
            df['x1'] = df['centerX'].diff(i)
            df['y1'] = df['centerY'].diff(i)
            df['T'] = df['x1'] ** 2 + df['y1'] ** 2
            msd_result.append([i * 0.06, df['T'].mean()])
        msd_df = pd.DataFrame(msd_result)
        msd_df.columns = ['t', 'msd']
        return msd_df


class TrackAnalyse(CSVCache):
    def __init__(self, *args, **kwargs):
        super(TrackAnalyse, self).__init__(*args, **kwargs)

    def execute(self, contains, particles, debug=False, *args, **kwargs):
        cx, cy, r = contains.values[0]

        def cul_dis(x, y, _x, _y):
            return math.sqrt((x - _x) * (x - _x) + (y - _y) * (y - _y))

        def cul_par_dis(row):
            x, y = row['centerX'], row['centerX']
            # cx, cy
            index = int(row['background_uid'] - 1)
            if index > len(contains):
                index = 0
            _cx, _cy, _cr = contains.values[index]
            return cul_dis(x, y, _cx, _cy)

        # 过滤容器外部的颗粒
        particles['dis'] = particles.apply(lambda x: cul_par_dis(x), axis=1)
        particles = particles[particles['dis'] <= r * 1.02]

        # 过滤跳跃式颗粒
        pre_line = None
        result = []
        for line in json.loads(particles.to_json(orient='records')):
            if pre_line is None:
                pre_line = line
                result.append(line)
                continue
            dis = cul_dis(line['centerX'], line['centerY'], pre_line['centerX'], pre_line['centerY'])
            if dis < 100:
                pre_line = line
                result.append(line)
        return pd.DataFrame(result)
