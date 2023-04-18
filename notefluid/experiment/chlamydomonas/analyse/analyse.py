import json
import math

import pandas as pd
from tqdm import tqdm

from notefluid.common.base.cache import CSVDataFrameCache
from notefluid.experiment.chlamydomonas.base.base import VideoBase


class TrackAnalyse(CSVDataFrameCache):
    def __init__(self, config: VideoBase, *args, **kwargs):
        self.config = config
        super(TrackAnalyse, self).__init__(filepath=f'{self.config.cache_dir}/particle_track.csv', *args, **kwargs)

    def _execute(self, contains, particles, debug=False, *args, **kwargs):
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
        # 过滤一帧两个颗粒的点
        result2 = []
        index = 0
        pre_line = None
        while index < len(result):
            if pre_line is None:
                pre_line = result[index]
                result2.append(result[index])
                continue
            index += 1
            if index >= len(result):
                break
            current_step = result[index]['step']
            index2 = index
            for i in range(10):
                if index2 >= len(result):
                    index2 -= 1
                    break
                if result[index2]['step'] > current_step:
                    index2 -= 1
                    break
                index2 += 1
            if index2 > index:
                max_i = index
                min_d = 10000.
                for i in range(index, index2 + 1):
                    dis = cul_dis(result[i]['centerX'], result[i]['centerY'], pre_line['centerX'], pre_line['centerY'])
                    if dis < min_d:
                        min_d = dis
                        max_i = i
                # print(index, index2, min_d)
                result2.append(result[max_i])
                index = index2
            else:
                result2.append(result[index])
            pre_line = result2[len(result2) - 1]
        return pd.DataFrame(result2)


class MSDCalculate(CSVDataFrameCache):
    def __init__(self, config: VideoBase, *args, **kwargs):
        self.config = config
        super(MSDCalculate, self).__init__(filepath=f'{self.config.cache_dir}/particle_track_msd.csv', *args, **kwargs)

    def _execute(self, track_df, *args, **kwargs):
        msd_result = []
        if track_df is None:
            return None
        df_fill = pd.DataFrame([[i + 1] for i in range(track_df['step'].max())])
        df_fill.columns = ['step']
        df_fill = pd.merge(df_fill, track_df, on='step', how='left')

        for i in tqdm(range(1, len(df_fill))):
            df_fill['x1'] = df_fill['centerX'].diff(i)
            df_fill['y1'] = df_fill['centerY'].diff(i)
            df_fill['T'] = df_fill['x1'] ** 2 + df_fill['y1'] ** 2
            msd_result.append([i * 0.06, df_fill['T'].sum(), df_fill['T'].count()])
        msd_df = pd.DataFrame(msd_result)
        msd_df.columns = ['t', 'sum', 'cnt']
        return msd_df
