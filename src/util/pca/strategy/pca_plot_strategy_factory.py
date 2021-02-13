from util.pca.strategy.pca_2d_plot_strategy import PCA2DPlotStrategy
from util.pca.strategy.pca_3d_plot_strategy import PCA3DPlotStrategy


class LowDimensionalDataException(Exception):
    def __init__(self, label):
        if label:
            m = f'Data must have at lest 3 dimensions 2 for feature columns and one for labels'
        else:
            m = f'Data must have at lest 2 dimensions.'
        super().__init__(m)


class PCAPlotStrategyFactory:
    @staticmethod
    def create(cfg):
        dimensions = cfg.get_dimensions()

        if dimensions < 2:
            raise LowDimensionalDataException(cfg.get_label())
        elif dimensions == 2:
            return PCA2DPlotStrategy(cfg)
        elif dimensions == 3:
            return PCA3DPlotStrategy(cfg)
