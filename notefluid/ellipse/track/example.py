import math

import pandas as pd
from matplotlib import pyplot as plt

from notefluid.ellipse.track.track_plot2 import FlowBase, Track, EllipseBaseTrack


def _load(path, index=0):
    df = pd.read_csv(path, sep='\s+', header=None)
    cols = [f"c{i}" for i in df.columns]
    cols[0] = 'y'
    cols[1] = 'x'
    cols[4] = 'theta'
    df.columns = cols
    df['theta'] = (df['theta'] + 0.5) * math.pi
    df = df.reset_index(names='step')
    df['index'] = index
    return df


dfs = [_load('/Users/chen/workspace/chenflow/flow230421/cmake-build-debug/orientation0.dat', 0),
       _load('/Users/chen/workspace/chenflow/flow230421/cmake-build-debug/orientation1.dat', 1)]
dfs = pd.concat(dfs)
dfs = dfs.sort_values('step')

track = Track()
track.set_flow(FlowBase(dfs['x'].max() + 10, dfs['y'].max() + 10))
track.add_ellipse(0, EllipseBaseTrack(4, 8))
track.add_ellipse(1, EllipseBaseTrack(4, 8))
track.load_ellipse(dfs, step=1)

plt.pause(10)
plt.show()
