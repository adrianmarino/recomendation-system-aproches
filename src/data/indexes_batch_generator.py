import numpy as np


class IndexesBatchGenerator:
    def __init__(self, examples_count, batch_size):
        self.indexes = np.arange(examples_count)
        self.batch_size = batch_size
        self.batches_count = int(np.floor(examples_count / batch_size))

    def batch(self, index, shuffle=False):
        if shuffle:
            np.random.shuffle(self.indexes)

        from_index = index * self.batch_size
        to_index = (index + 1) * self.batch_size
        return self.indexes[from_index:to_index]
