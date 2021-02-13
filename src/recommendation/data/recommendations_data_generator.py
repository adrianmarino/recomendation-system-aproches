import numpy as np

from data.generator.data_frame.data_frame_data_generatror import DataFrameDataGenerator


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
        embeddings = np.array(
            page.select(self.__column_manager.emb_features) \
                .rdd \
                .map(tuple) \
                .collect()
        )

        genres = np.array(
            page.select(self.__column_manager.gender_features) \
                .rdd \
                .map(lambda t: np.array(t)) \
                .collect()
        )

        return [embeddings[:, 0], embeddings[:, 1], genres]
