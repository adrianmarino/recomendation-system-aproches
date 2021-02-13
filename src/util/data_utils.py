from tensorflow.data import Dataset


def dataframe_to_dataset(data_frame, label, shuffle=False, batch_size=0):
    """
    Create a Tensorflow Dataset from a Pandas DataFrame

    :param data_frame: a panda data frame
    :param  label: label column name
    :param shuffle: makes aleatory roes order
    :param batch_size: split rows in N batches ir size equals to batch_size.
    :return: a Dataset.
    """

    feature_columns = data_frame.copy()
    label_column = feature_columns.pop(label)

    data_set = Dataset.from_tensor_slices((
        dict(feature_columns),
        label_column
    ))

    if shuffle:
        data_set = data_set.shuffle(buffer_size=len(feature_columns))

    return data_set.batch(batch_size) if batch_size > 0 else data_set
