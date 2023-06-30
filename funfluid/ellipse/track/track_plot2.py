# -*- coding: utf-8 -*-
import math
from typing import List

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


class EllipseTrack:
    def __init__(self, df, a=10, b=5, color=None, marker=None, *args, **kwargs):
        self.df = df
        self.a = a
        self.b = b
        self.color = color
        self.marker = marker
        self.snapshot_steps = []
        self.lns = []

    def transform(self):
        self.df['xx'] = self.df['x']
        self.df['x'] = self.df['y']
        self.df['y'] = self.df['xx']
        self.df['theta'] = self.df['theta'] - math.pi / 2

    @property
    def min_x(self):
        return self.df['x'].min()

    @property
    def max_x(self):
        return self.df['x'].max()

    @property
    def min_y(self):
        return self.df['y'].min()

    @property
    def max_y(self):
        return self.df['y'].max()

    @property
    def max_step(self):
        return self.df['step'].max()

    def add_snapshot(self, step=0, color=None, marker=None):
        self.snapshot_steps.append({
            "step": step,
            "color": color or self.color,
            "marker": marker or self.marker
        })

    def plot_ref(self, line_width=0.3):
        self.lns = []
        self.lns.append(plt.plot([], [], color=self.color, marker=self.marker, linewidth=line_width, alpha=0.8)[0])
        self.lns.append(plt.plot([], [], color=self.color, marker=self.marker, linewidth=line_width)[0])

        for i, record in enumerate(self.snapshot_steps):
            self.lns.append(plt.plot([], [], color=record['color'], marker=record['marker'], linewidth=line_width)[0])
        return self.lns

    def _get_ellipse_data(self, step):
        x0 = self.df['x'][step]
        y0 = self.df['y'][step]
        theta = self.df['theta'][step]
        phi = np.array([i / 100. * np.pi for i in range(201)])
        x = np.cos(theta) * self.a * np.cos(phi) - np.sin(theta) * self.b * np.sin(phi) + x0
        y = np.sin(theta) * self.a * np.cos(phi) + np.cos(theta) * self.b * np.sin(phi) + y0
        return x, y

    def plot_update(self, step):
        self.lns[0].set_data(self.df['x'][:step], self.df['y'][:step])
        self.lns[1].set_data(*self._get_ellipse_data(step=step))

        for i, record in enumerate(self.snapshot_steps):
            if record['step'] <= step:
                self.lns[i + 2].set_data(*self._get_ellipse_data(record['step']))

        return self.lns

    def plot(self, step=10):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1200)
        ax.set_aspect(1)
        self.plot_ref()
        ani = animation.FuncAnimation(fig=fig,
                                      func=self.plot_update,
                                      frames=[i for i in range(2, self.df['step'].max() - 2, step)],
                                      interval=100,
                                      # init_func=self.plot_ref,
                                      blit=True,
                                      repeat=False
                                      )

        plt.show()
        # ani.save("a.gif", writer='imagemagick')


class Tracks:
    def __init__(self, flow: FlowBase = None):
        self.flow = flow
        self.ellipses: List[EllipseTrack] = []
        self.lns = []

    def set_flow(self, flow):
        self.flow = flow

    def transform(self):
        for ellipse in self.ellipses:
            ellipse.transform()

    @property
    def max_x(self):
        return max([ellipse.max_x for ellipse in self.ellipses])

    @property
    def min_x(self):
        return min([ellipse.min_x for ellipse in self.ellipses])

    @property
    def max_y(self):
        return max([ellipse.max_y for ellipse in self.ellipses])

    @property
    def min_y(self):
        return min([ellipse.min_y for ellipse in self.ellipses])

    @property
    def max_step(self):
        steps = 100
        for ellipse in self.ellipses:
            steps = max(steps, ellipse.max_step)
        return steps

    def add_ellipse(self, ellipse: EllipseTrack):
        self.ellipses.append(ellipse)

    def plot_ref(self, ax):
        self.flow.figure(ax)
        for ellipse in self.ellipses:
            self.lns.extend(ellipse.plot_ref())

    def plot_update(self, step=10):
        for ellipse in self.ellipses:
            ellipse.plot_update(step=step)
        self.lns[-1].set_text(f'step={step}')
        return self.lns

    def plot(self, step=10, gif_path='./trak.gif'):
        fig, ax = plt.subplots()

        plt.grid(ls='--')
        plt.xlabel(r'$x/L$')
        plt.ylabel(r'$y/L$')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1200)
        ax.set_aspect(1)
        plt.tick_params(labelsize=11)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        self.plot_ref(ax)
        print(self.lns)
        self.lns.append(plt.title('', fontsize=12))
        ani = animation.FuncAnimation(fig=fig,
                                      func=self.plot_update,
                                      frames=[i for i in range(2, self.max_step, step)],
                                      interval=100,
                                      blit=False,
                                      repeat=False
                                      )

        plt.show()
        ani.save(gif_path, writer='imagemagick')
