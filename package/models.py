from tensorflow import keras
from tensorflow.keras import layers, regularizers


### NOTE - CURRENTLY USES LOCAL FILES, NOT DB
### TODO - REFACTOR FOR DATABASE USAGE


def dnn(lookback, n_features, n_out):
    inputs = keras.Input(shape=(lookback, n_features), name='in1')
    x = layers.Dense(units=64, kernel_regularizer=regularizers.l2(.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Dropout(rate=.4)(x)

    x = layers.Dense(units=64, kernel_regularizer=regularizers.l2(.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Dropout(rate=.4)(x)

    x = layers.Dense(units=32, kernel_regularizer=regularizers.l2(.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Dropout(rate=.4)(x)

    outputs = layers.Dense(n_out)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def cnn(lookback, n_features, n_out):
    inputs = keras.Input(shape=(lookback, n_features), name='in1')

    x = layers.Conv1D(filters=n_features *3, kernel_size=5, strides=1, padding='same', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(.01))(inputs)
    # x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)

    x = layers.Conv1D(filters=n_features *2, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(.01))(x)
    # x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)

    x = layers.Conv1D(filters=1, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(.01))(x)
    # x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)

    x = layers.Flatten()(x)

    x = layers.Dense(units=64, activation='relu', kernel_regularizer=regularizers.l2(.01))(x)
    x = layers.Dropout(rate=.25)(x)

    x = layers.Dense(units=32, activation='relu', kernel_regularizer=regularizers.l2(.01))(x)
    x = layers.Dropout(rate=.25)(x)

    outputs = layers.Dense(n_out)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


def lstm(lookback: int = 5,
         n_features: int = None,
         n_out: int = 1,
         return_layers: int = 2,
         units: int = 64,
         dropout: float = .25,
         rec_dropout: float = .25,
         l1_reg: float = .01,
         l2_reg: float = .01,
         l1l2_reg: float = .01) -> keras.Model:
    assert n_features is not None, '[ERROR] must supply n_features (int)'

    inputs = keras.Input(shape=(lookback, n_features), name='in1')

    x = layers.LSTM(units=64, return_sequences=True, recurrent_dropout=.25, kernel_regularizer=regularizers.l2(.01))(inputs)
    x = layers.Dropout(rate=.25)(x)

    x = layers.LSTM(units=64, return_sequences=False, recurrent_dropout=.25, kernel_regularizer=regularizers.l2(.01))(x)
    x = layers.Dropout(rate=.25)(x)

    x = layers.Dense(units=64, activation='relu', kernel_regularizer=regularizers.l2(.01))(x)
    x = layers.Dropout(rate=.25)(x)

    outputs = layers.Dense(units=n_out)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


