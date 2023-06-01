# -*- coding: utf-8 -*-
from typing import Dict

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

    def plot(self, color='g'):
        theta = np.array([i / 100. * np.pi for i in range(201)])
        x = np.cos(self.theta) * self.a * np.cos(theta) - np.sin(self.theta) * self.b * np.sin(theta) + self.x0
        y = np.sin(self.theta) * self.a * np.cos(theta) + np.cos(self.theta) * self.b * np.sin(theta) + self.y0
        plt.plot(y, x, color)
        return x, y


class EllipseBaseTrack(EllipseBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.track = []

    def update(self, x0=0, y0=0, theta=0):
        self.track.append([x0, y0, theta])
        super().update(x0=x0, y0=y0, theta=theta)

    def plot(self, color='g'):
        super().plot(color=color)
        track = np.array(self.track)
        if len(track) > 1:
            plt.plot(track[:, 0], track[:, 1], color)


class Track:
    def __init__(self, flow: FlowBase = None):
        self.flow = flow
        self.ellipse_list: Dict[int, EllipseBaseTrack] = {}

    def set_flow(self, flow):
        self.flow = flow

    def add_ellipse(self, index, ellipse: EllipseBaseTrack):
        self.ellipse_list[index] = ellipse

    def load_ellipse(self, df, debug=True):
        if debug:
            plt.figure(figsize=[16, 4])
            plt.ion()

        def load_row(row):
            index = row['index']
            if index not in self.ellipse_list.keys():
                print(f"{index} is not in {self.ellipse_list.keys()}")
            ellipse = self.ellipse_list[index]
            ellipse.update(row['x'], row['y'], row['theta'])
            if debug:
                plt.cla()
                plt.title(f"step={row['step']}")
                # plt.axis([0, 800, 0, 200])
                self.flow.plot()
                for ellipse in self.ellipse_list.values():
                    ellipse.plot(color='g')
                plt.pause(0.2)

        df.apply(lambda x: load_row(x), axis=1)

        if debug:
            plt.ioff()
            plt.pause(1)
            plt.show()
