import collections

from tensorflow.keras.layers import DenseFeatures, Activation, Lambda, Dense, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

Layer = collections.namedtuple('Layer', 'units act dropout')


class WideAndDeepModelFactory:
    @staticmethod
    def create(
            wide_input_setttings,
            deep_input_setttings,
            deep_hidden_layers,
            max_rating,
            min_rating,
            lr,
            loss='mean_squared_error'
    ):
        wide = DenseFeatures(
            wide_input_setttings.feature_columns(),
            name='Wide_Features'
        )(wide_input_setttings.inputs_dic())
        # wide = Dropout(0.2)(wide)

        deep = DenseFeatures(
            deep_input_setttings.feature_columns(),
            name='Deep_Features'
        )(deep_input_setttings.inputs_dic())

        for idx, layer in enumerate(deep_hidden_layers):
            layer_name = f'Deep_Hidden_{idx + 1}'
            deep = Dense(name=layer_name, units=layer.units, activation=layer.act)(deep)
            deep = Dropout(layer.dropout)(deep)

        both = concatenate([deep, wide], name='Deep_Wide')

        output = Dense(1, kernel_initializer='he_normal')(both)
        output = Activation('sigmoid')(output)
        output = Lambda(lambda x: x * (max_rating - min_rating) + min_rating, name='user_rating_prediction')(output)

        inputs = list(wide_input_setttings.inputs()) + list(deep_input_setttings.inputs())

        model = Model(inputs, output, name='Wide_And_Deep_Model')

        model.compile(optimizer=Adam(lr), loss=loss)

        return model
