from data.generator.data_frame.data_frame_data_generatror import DataFrameDataGenerator
from spark import get_rows
from util.profiler import Profiler


class RecommendationsDataGenerator(DataFrameDataGenerator):
    def __init__(
            self,
            data_frame,
            batch_size,
            column_manager,
            shuffle=True,
            name='data-frame'
    ):
        super().__init__(
            data_frame,
            column_manager.features,
            column_manager.labels,
            batch_size,
            shuffle,
            name
        )
        self.__column_manager = column_manager

    def _features(self, page, columns):
        with Profiler('Resolve users embedding'):
            emb1 = get_rows(page, self.__column_manager.emb_features[0])

        with Profiler('Resolve movies embedding'):
            emb2 = get_rows(page, self.__column_manager.emb_features[1])

        with Profiler('Resolve genders one-hot array'):
            genders = get_rows(page, self.__column_manager.gender_features)
        return [emb1, emb2, genders]
