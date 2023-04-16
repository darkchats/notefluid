# -*- coding: utf-8 -*-
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


# scatter_plot()

class Track:
    def __init__(self, fiber1_path=None, fiber2_path=None):
        self.fiber1_path = fiber1_path or '/Users/chen/workspace/chenflow/paricles/cmake-build-debug/orientation0.dat'
        self.fiber2_path = fiber2_path or '/Users/chen/workspace/chenflow/paricles/cmake-build-debug/orientation1.dat'
        self.fiber1 = None
        self.fiber2 = None

    def load(self):
        def _load(path):
            df = pd.read_csv(path, sep='\s+', header=None)
            cols = [f"c{i}" for i in df.columns]
            cols[0] = 'x'
            cols[1] = 'y'
            cols[4] = 'theta'
            df.columns = cols
            df['theta'] = (df['theta'] + 0.5) * math.pi
            return df

        self.fiber1 = _load(self.fiber1_path)
        self.fiber2 = _load(self.fiber2_path)

        print(self.fiber1)

    def transform(self, x0, y0, a, b, phi):
        theta = np.array([i / 100. * np.pi for i in range(201)])
        x = np.cos(phi) * a * np.cos(theta) - np.sin(phi) * b * np.sin(theta) + x0
        y = np.sin(phi) * a * np.cos(theta) + np.cos(phi) * b * np.sin(theta) + y0
        return x, y

    def step_plot(self, step):
        # 打开交互模式

        x1, y1 = self.transform(self.fiber1['x'][step], self.fiber1['y'][step], 10, 5, self.fiber1['theta'][step])
        x2, y2 = self.transform(self.fiber2['x'][step], self.fiber2['y'][step], 10, 5, self.fiber2['theta'][step])

        plt.plot(y1, x1, 'g')
        plt.plot(y2, x2, 'r')

        plt.plot(self.fiber1['y'][:step], self.fiber1['x'][:step], 'g')
        plt.plot(self.fiber2['y'][:step], self.fiber2['x'][:step], 'r')

    def plot(self):
        # 打开交互模式

        plt.figure(figsize=[16, 4])
        plt.ion()

        pbr = tqdm(range(500))
        for step in pbr:
            try:
                bbb = 20
                if (step * bbb > len(self.fiber1)):
                    plt.pause(1000)
                plt.cla()
                plt.title(f"step={step * 5 * bbb}")
                plt.axis([0, 800, 0, 200])
                plt.grid(True)

                self.step_plot(step * bbb)

                plt.pause(0.2)
            except Exception as e:
                break

        plt.ioff()
        plt.pause(2)
        plt.show()

    def run(self):
        self.load()
        self.plot()


Track().run()
plt.show()
