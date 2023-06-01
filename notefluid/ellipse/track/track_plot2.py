# -*- coding: utf-8 -*-
from typing import Dict

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


class FlowBase:
    def __init__(self, weight=800, height=200, x_start=0, y_start=0):
        self.weight = weight
        self.height = height
        self.x_start = x_start
        self.y_start = y_start

    def plot(self):
        plt.axis([self.x_start, self.weight, self.y_start, self.height])
        plt.grid(True)

    def figure(self, ax):
        ax.set_xlim(0, self.weight)
        ax.set_ylim(0, self.height)
        ax.set_aspect(1)


class EllipseBase:
    def __init__(self, a=1, b=1):
        self.a = a
        self.b = b
        self.theta = 0
        self.x0 = 0
        self.y0 = 0

    def update(self, x0=0, y0=0, theta=0):
        self.x0 = x0
        self.y0 = y0
        self.theta = theta

    def plot_data(self, x0=0, y0=0, theta=0):
        phi = np.array([i / 100. * np.pi for i in range(201)])
        x = np.cos(theta) * self.a * np.cos(phi) - np.sin(theta) * self.b * np.sin(phi) + x0
        y = np.sin(theta) * self.a * np.cos(phi) + np.cos(theta) * self.b * np.sin(phi) + y0
        return x, y


class EllipseBaseTrack(EllipseBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.track = []

    def update(self, x0=0, y0=0, theta=0):
        self.track.append([x0, y0, theta])
        super().update(x0=x0, y0=y0, theta=theta)

    def plot_data(self, x0=0, y0=0, theta=0, step=0):
        return super().plot_data(self.track[step][0], self.track[step][1], self.track[step][2])


class Track:
    def __init__(self, flow: FlowBase = None):
        self.flow = flow
        self.ellipse_list: Dict[int, EllipseBaseTrack] = {}

    def set_flow(self, flow):
        self.flow = flow

    def add_ellipse(self, index, ellipse: EllipseBaseTrack):
        self.ellipse_list[index] = ellipse

    def load_ellipse(self, df, step=10, gif_path='result.gif', colors=None, line_width=0.3):
        fig, ax = plt.subplots()

        lns = []
        colors = colors or ['r', 'b', 'b', 'b', 'b', 'b']
        for i, _ in enumerate(self.ellipse_list):
            ln1, = plt.plot([], [], f'{colors[i]}-', linewidth=line_width, animated=True)
            ln2, = plt.plot([], [], f'{colors[i]}-', linewidth=line_width, animated=True)
            lns.append(ln1)
            lns.append(ln2)

        def load_row(row):
            index = row['index']
            if index not in self.ellipse_list.keys():
                print(f"{index} is not in {self.ellipse_list.keys()}")
            ellipse = self.ellipse_list[index]
            ellipse.update(row['x'], row['y'], row['theta'])

        df.apply(lambda x: load_row(x), axis=1)

        def init():
            self.flow.figure(ax)
            return *lns,

        def update(step):
            for i, ellipse in enumerate(self.ellipse_list.values()):
                track = np.array(ellipse.track)
                lns[2 * i].set_data(track[:step, 0], track[:step, 1])
                lns[2 * i + 1].set_data(*ellipse.plot_data(step=step))
            return *lns,

        ani = animation.FuncAnimation(fig, update, frames=[i for i in range(2, df['step'].max() - 2, step)],
                                      interval=10,
                                      init_func=init,
                                      blit=True, repeat=False)
        plt.show()
        ani.save(gif_path, writer='imagemagick')
