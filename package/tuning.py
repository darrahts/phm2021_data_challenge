import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

from kerastuner_tensorboard_logger import (
    TensorBoardLogger,
    setup_tb  # Optional
)


class Tuning():
    """
        @brief: current values are hardcoded, should be parameterized

        @functions:
            create_hypermodel: the function passed to the keras_tuner.Tuner
            build_model: builds the model with the specified parameters
    """

    input_shape = -1
    num_outputs = -1

    def __init__(self, input_shape, num_outputs, *args, **kwargs):
        self.input_shape = input_shape
        self.num_outputs = num_outputs

        x = 1 if not 'a' in kwargs else kwargs['a']


    def create_hypermodel(self, hp):
        """
            @brief: the hypermodel function passed to the tuner, calls build_model
        """
        max_layers = 4
        # tune the number of layers
        layers = hp.Int('layers', min_value=2, max_value=max_layers, step=1)

        # tune the number of units per layer
        units = []
        for i in range(max_layers):
            units.append(hp.Int(f'units_{i}', min_value=32, max_value=512, step=32))

        # tune the dropout rate
        dropout_rate = hp.Choice('dropout_rate', values=[.2, .25, .4])

        # tune regularization rate
        l2 = hp.Choice('l2', values=[.0001, .00025, .0005, .001])

        # tune the learning rate for the optimizer
        learning_rate = hp.Choice('learning_rate', values=[.0001, .00025, .0005, .001, .0025])

        params = MyParameters(layers=layers, units=units, dropout_rate=dropout_rate, l2=l2, learning_rate=learning_rate)

        ############################
        ########## model ###########
        ############################

        return self.build_model(params)


    def build_model(self, params, *args, **kwargs):
        """
            @brief: builds the model with the specified params
        """
        model = keras.Sequential()

        model.add(keras.Input(shape=self.input_shape))

        for i in range(params.layers):
            model.add(keras.layers.Dense(units=params.units[i],
                                         activation='relu',
                                         activity_regularizer=keras.regularizers.l2(l2=params.l2),
                                         name=f'hidden_{i}'))

            model.add(keras.layers.Dropout(rate=params.dropout_rate))

        model.add(keras.layers.Dense(self.num_outputs))

        # compile
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=params.learning_rate),
                      loss='mse',
                      metrics=[keras.metrics.RootMeanSquaredError()])

        return model


###############################################################################################################
###############################################################################################################

class MyTuner(kt.Tuner):
    """
        @brief: overrides built-in

        @functions:
            run_trial: allows for batch_size to be tuned
    """
    def run_trial(self, trial, *args, **kwargs):
        min_batch_size = 32 if not 'min_batch_size' in kwargs else kwargs['min_batch_size']
        max_batch_size = 256 if not 'max_batch_size' in kwargs else kwargs['max_batch_size']
        batch_size_step = 32 if not 'batch_size_step' in kwargs else kwargs['max_batch_size']

        kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', min_batch_size, max_batch_size, step=batch_size_step)
        return super(MyTuner, self).run_trial(trial, *args, **kwargs)


###############################################################################################################
###############################################################################################################


class MyParameters():
    """
        @brief: container class to hold sets of parameters
    """
    def __init__(self, *args, **kwargs):
        self.layers = kwargs['layers']
        self.units = kwargs['units']
        self.dropout_rate = kwargs['dropout_rate']
        self.l2 = kwargs['l2']
        self.learning_rate = kwargs['learning_rate']
        x = 1 if not 'a' in kwargs else kwargs['a']
        self.metric = "na" if not 'metric' in kwargs else kwargs['metric']
        self.score = -1 if not 'score' in kwargs else kwargs['score']

    def __str__(self):
        return self.layers















