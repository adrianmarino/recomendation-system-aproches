import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

from util.pca.strategy.pca_plot_strategy import PCAPlotStrategy


class PCA3DPlotStrategy(PCAPlotStrategy):
    def _create_canvas(self, figure):
        ax = Axes3D(
            figure,
            rect=[0, 0, .95, 1],
            elev=self.cfg.get_elevation(),
            azim=self.cfg.get_azimuthal_angle()
        )
        return ax

    def _setup_axis_labels(self, ax):
        ax.set_xlabel(self.cfg.get_axis_names()[0], fontsize=self.cfg.get_axis_font_size())
        ax.set_ylabel(self.cfg.get_axis_names()[1], fontsize=self.cfg.get_axis_font_size())
        ax.set_zlabel(self.cfg.get_axis_names()[2], fontsize=self.cfg.get_axis_font_size())

    def _scatter(self, ax, data, color, label=None):
        x_set, y_set, z_set = data[:, 0], data[:, 1], data[:, 2]

        ax.scatter(
            x_set,
            y_set,
            z_set,
            color=color,
            label=label,
            s=self.cfg.get_point_radius(),
            cmap=cm.get_cmap("Spectral"),
            edgecolor='k'
        )

        if self.cfg.get_show_annotations() and label:
            [ax.text(x, y, z, label, fontsize=self.cfg.get_annotations_font_size()) for (x, y, z) in zip(x_set, y_set, z_set)]
