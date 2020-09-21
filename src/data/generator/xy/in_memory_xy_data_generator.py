from data.generator.xy.abstract_xy_data_generator import AbstractXyDataGenerator


class InMemoryXyDataGenerator(AbstractXyDataGenerator):
    def __init__(self, X, y, batch_size=None, shuffle=False, to_input=lambda X, y: (X, y)):
        self.X, self.y = X, y
        super().__init__(len(X), batch_size, shuffle),
        self.__to_input = to_input

    def _load_batch(self, batch_elements_indexes):
        X, y = self.X[batch_elements_indexes], self.y[batch_elements_indexes]
        return self.__to_input(X, y)
