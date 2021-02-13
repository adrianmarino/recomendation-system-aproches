from tensorflow import feature_column

fc = feature_column


class FeatureColumn:
    @staticmethod
    def cat_id(name, ids_count, unknown_value):
        return fc.indicator_column(fc.categorical_column_with_identity(
            key=name,
            num_buckets=ids_count,
            default_value=unknown_value
        ))

    @staticmethod
    def num(name): return fc.numeric_column(name)

    @staticmethod
    def num_bucketized(name, boundaries):
        return fc.bucketized_column(FeatureColumn.num(name), boundaries=boundaries)

    @staticmethod
    def cat(name, values):
        return fc.categorical_column_with_vocabulary_list(name, values)

    @staticmethod
    def cat_one_hot(name, values):
        return fc.indicator_column(FeatureColumn.cat(name, values))

    @staticmethod
    def cat_one_hot_crossed(columns, hash_bucket_size=1000):
        return fc.indicator_column(fc.crossed_column(columns, hash_bucket_size=hash_bucket_size))

    @staticmethod
    def cat_embedding(name, values, dimension=8):
        return fc.embedding_column(FeatureColumn.cat(name, values), dimension=dimension)
