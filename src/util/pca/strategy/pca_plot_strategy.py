import abc

import matplotlib.pyplot as plt
import numpy as np

from util.plot import random_colors


class PCAPlotStrategy(abc.ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    def plot(self):
        figure = plt.figure(figsize=self.cfg.get_figure_size())
        plt.clf()

        ax = self._create_canvas(figure)

        ax.set_title(self.cfg.get_title(), fontsize=self.cfg.get_title_font_size())

        self._setup_axis_labels(ax)

        if self.cfg.get_labels():
            labels = labels = self.cfg.get_labels()
            labels_color = random_colors(len(labels), distance=self.cfg.get_point_color_distance())

            for label, color in zip(labels.keys(), labels_color):
                indices = labels[label]
                data = np.take(self.cfg.get_data(), axis=0, indices=indices)
                self._scatter(ax, data, color, label)

            if self.cfg.get_show_legend():
                plt.legend()
        else:
            self._scatter(ax, self.cfg.get_data(), self.cfg.get_point_color())

        ax.grid()

    def __data_by_label(self, label):
        return self.cfg.get_data()[self.cfg.get_data()[self.cfg.get_label()] == label]

    @abc.abstractmethod
    def _create_canvas(self, figure):
        pass

    @abc.abstractmethod
    def _setup_axis_labels(self, ax, data):
        pass

    @abc.abstractmethod
    def _scatter(self, ax, data, color, label=None):
        pass
