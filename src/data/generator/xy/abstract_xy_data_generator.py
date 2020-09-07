import abc

from tensorflow.keras.utils import Sequence

from data.indexes_batch_generator import IndexesBatchGenerator


class AbstractXyDataGenerator(Sequence):
    def __init__(self, examples_count, batch_size=None, shuffle=True):
        self.shuffle = shuffle
        self.__indexes_batch_generator = IndexesBatchGenerator(
            examples_count,
            examples_count if batch_size is None else batch_size
        )

    def __len__(self):
        return self.__indexes_batch_generator.batches_count

    def __getitem__(self, batch_index):
        batch_elements_indexes = self.__indexes_batch_generator.batch(batch_index, self.shuffle)
        X, y = self._load_batch(batch_elements_indexes)
        return X, y

    @abc.abstractmethod
    def _load_batch(self, batch_elements_indexes):
        pass
