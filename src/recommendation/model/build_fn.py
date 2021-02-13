from tensorflow.keras.layers import Input, Activation, Lambda, Dense, Dropout

from model.embedding.embedding_layer import EmbeddingLayer


def embedding_input(n_items, n_min_factors, name, shape=(1,), l2_delta=1e-6):
    input = Input(shape=shape, name=f'{name}_idx')

    emb = EmbeddingLayer(
        n_items=n_items,
        n_min_factors=n_min_factors,
        l2_delta=l2_delta,
        name=f'{name}_embedding'
    )(input)

    return input, emb


def biased_embedding_input(n_items, n_min_factors, name, shape=(1,), l2_delta=1e-6):
    input = Input(shape=shape, name=f'{name}_idx')

    emb = EmbeddingLayer(
        n_items=n_items,
        n_min_factors=n_min_factors,
        l2_delta=l2_delta,
        name=f'{name}_embedding')(input)

    bias_emb = EmbeddingLayer(n_items=n_items, n_factors=1, name=f'{name}_biases')(input)

    return input, emb, bias_emb


def dense_layers(net, units, dropout, min_rating, max_rating):
    for layer_units, layer_dropout in zip(units, dropout):
        net = Dense(layer_units, kernel_initializer='he_normal')(net)
        net = Activation('relu')(net)
        if layer_dropout > 0:
            net = Dropout(layer_dropout)(net)

    net = Dense(1, kernel_initializer='he_normal')(net)
    net = Activation('sigmoid', name='sigmoid_activation')(net)
    return Lambda(lambda p: p * (max_rating - min_rating) + min_rating, name='user_rating_prediction')(net)
