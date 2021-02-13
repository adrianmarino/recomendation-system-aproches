from util.pca.strategy.pca_plot_strategy import PCAPlotStrategy


class PCA2DPlotStrategy(PCAPlotStrategy):
    def _create_canvas(self, figure):
        return figure.add_subplot(1, 1, 1)

    def _setup_axis_labels(self, ax):
        ax.set_xlabel(self.cfg.get_axis_names()[0], fontsize=self.cfg.get_axis_font_size())
        ax.set_ylabel(self.cfg.get_axis_names()[1], fontsize=self.cfg.get_axis_font_size())

    def _scatter(self, ax, data, color, label=None):
        x_set, y_set = data[:, 0], data[:, 1]

        ax.scatter(x_set, y_set, label=label, color=color, s=self.cfg.get_point_radius())

        if self.cfg.get_show_annotations() and label:
            [ax.annotate(label, point, fontsize=self.cfg.get_annotations_font_size()) for point in zip(x_set, y_set)]
