import numpy as np

from util.pca.strategy.pca_plot_strategy_factory import PCAPlotStrategyFactory
from util.plot import random_color


class PCAPlotter:
    def __init__(self):
        self.canvas(size=(8, 8), elevation=20, azimuthal_angle=60) \
            .axis_names(['X', 'Y', 'Z']) \
            .title('PCA') \
            .labels({}) \
            .point(radius=25) \
            .annotations(True) \
            .legend(False)

    def data(self, value):
        self.__data = np.array(value)
        return self

    def canvas(self, size, elevation=20, azimuthal_angle=60):
        self.__figure_size = size
        self.__elevation = elevation
        self.__azimuthal_angle = azimuthal_angle
        return self

    def labels(self, labels):
        self.__labels = labels
        return self

    def axis_names(self, names, font_size=15):
        self.__axis_names = names
        self.__axis_font_size = font_size
        return self

    def title(self, title, font_size=20):
        self.__title = title
        self.__title_font_size = font_size
        return self

    def point(self, color=None, color_distance=0.4, radius=25):
        if not color:
            color = random_color()
        self.__point_color = color
        self.__point_color_distance = color_distance
        self.__point_radius = radius
        return self

    def annotations(self, show=True, font_size=15):
        self.__show_annotations = show
        self.__annotations_font_size = font_size
        return self

    def legend(self, value=True):
        self.__show_legend = value
        return self

    def plot(self):
        PCAPlotStrategyFactory.create(self).plot()

    def get_dimensions(self):
        return self.get_data().shape[1]

    def get_data(self): return self.__data

    def get_figure_size(self): return self.__figure_size

    def get_axis_names(self): return self.__axis_names

    def get_axis_font_size(self): return self.__axis_font_size

    def get_title(self): return self.__title

    def get_title_font_size(self): return self.__title_font_size

    def get_point_radius(self): return self.__point_radius

    def get_point_color(self): return self.__point_color

    def get_point_color_distance(self): return self.__point_color_distance

    def get_elevation(self): return self.__elevation

    def get_azimuthal_angle(self): return self.__azimuthal_angle

    def get_show_legend(self): return self.__show_legend

    def get_show_annotations(self): return self.__show_annotations

    def get_annotations_font_size(self): return self.__annotations_font_size

    def get_labels(self):
        return self.__labels
