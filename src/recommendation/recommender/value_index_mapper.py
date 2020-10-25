class ValueIndexMapper:
    def __init__(self, df, value_col, index_col):
        self.__value_2_index = {}
        self.__index_2_value = {}

        selected_df = df[[value_col, index_col]]
        selected_df = selected_df.drop_duplicates(keep='last')

        for _, row in selected_df.iterrows():
            self.__value_2_index[row[value_col]] = row[index_col]
            self.__index_2_value[row[index_col]] = row[value_col]

    def to_index(self, value):
        return self.__value_2_index[value]

    def to_value(self, index):
        return self.__index_2_value[index]

    def to_indexes(self, values):
        return [self.to_index(v) for v in values]

    def to_values(self, indexes):
        return [self.to_value(i) for i in indexes]
