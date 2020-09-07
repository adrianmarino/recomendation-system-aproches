from tensorflow.keras.utils import Sequence


class DataSetDataGenerator(Sequence):
    def __init__(self, data_set): self.__batches = list(data_set.as_numpy_iterator())

    def __len__(self): return len(self.__batches)

    def __getitem__(self, index): return self.__batches[index]
