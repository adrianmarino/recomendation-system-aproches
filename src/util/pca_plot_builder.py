import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

class PCAPlotBuilder:
    def __init__(self, data):
        self.__data = np.array(data)
        self.figure_size((8,8)) \
            .columns(['X', 'Y', 'Z']) \
            .title('PCA') \
            .point_radius(25) \
            .point_color('r') \
            .elevation(20) \
            .azimuthal_angle(60)

    def figure_size(self, figure_size):
        self.__figure_size = figure_size
        return self

    def columns(self, columns, font_size = 15):
        self.__columns = columns
        self.__label_font_size = font_size
        return self

    def title(self, title, font_size = 20):
        self.__title = title
        self.__title_font_size = font_size
        return self

    def point_radius(self, point_radius):
        self.__point_radius = point_radius
        return self

    def point_color(self, point_color):
        self.__point_color = point_color
        return self

    def elevation(self, value):
        self.__elevation = value
        return self
    
    def azimuthal_angle(self, value):
        self.__azimuthal_angle = value
        return self
    
    def __build_2d(self):
        figure = plt.figure(figsize = self.__figure_size)
        plt.clf()
        ax = figure.add_subplot(1,1,1)

        ax.set_title(self.__title, fontsize =  self.__title_font_size)
  
        ax.set_xlabel(self.__columns[0], fontsize = self.__label_font_size)
 
        ax.set_ylabel(self.__columns[1], fontsize = self.__label_font_size)

        ax.scatter(self.__data[:, 0], self.__data[:, 1], c = self.__point_color, s = self.__point_radius)
        ax.grid()

    def __build_3d(self):
        figure = plt.figure(figsize = self.__figure_size)
        plt.clf()
        ax = Axes3D(
            figure, 
            rect=[0, 0, .95, 1], 
            elev=self.__elevation, 
            azim=self.__azimuthal_angle
        )
        ax.set_title(self.__title, fontsize =  self.__title_font_size)
  
        ax.scatter(
            self.__data[:, 0], 
            self.__data[:, 1], 
            self.__data[:, 2], 
            c = self.__point_color,
            s = self.__point_radius,
            cmap = cm.get_cmap("Spectral"),
            edgecolor = 'k'
        )

        ax.set_xlabel(self.__columns[0], fontsize = self.__label_font_size)
        ax.set_ylabel(self.__columns[1], fontsize = self.__label_font_size)
        ax.set_zlabel(self.__columns[2], fontsize = self.__label_font_size)

    def build(self):
        if self.__data.shape[1] == 2:
            self.__build_2d()
        if self.__data.shape[1] == 3:
            self.__build_3d()