import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

import keras_tuner as kt
from keras import backend

import numpy as np

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




    @tf.function
    def loss_fcn(self, y_t, y_p):
        
        y_pred = tf.convert_to_tensor(y_p)
        y_true = tf.cast(y_t, y_pred.dtype)
        
        diff = y_pred-y_true
            
        res = tf.map_fn(fn=lambda x: tf.math.exp(-x/13) if x < 0 else tf.math.exp(x/10), elems=diff)
        s_score = tf.math.reduce_sum(res)

        rmse = tf.math.reduce_sum(backend.mean(tf.math.squared_difference(y_pred, y_true), axis=-1))

        score = s_score + rmse
    
        return score / 2.0

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
        # input layer
        inputs = keras.Input(shape=self.input_shape, name='in1')

        # first layer
        x = layers.Dense(units=params.units[0],
                         activation='relu',
                         activity_regularizer=keras.regularizers.l2(l2=params.l2),
                         name=f'hidden_{0}')(inputs)
        x = layers.Dropout(rate=params.dropout_rate)(x)

        # subsequent layers
        for i in range(1, params.layers):
            x = layers.Dense(units=params.units[i],
                             activation='relu',
                             activity_regularizer=keras.regularizers.l2(l2=params.l2),
                             name=f'hidden_{i}')(x)
            x = layers.Dropout(rate=params.dropout_rate)(x)

        # output layer
        outputs = layers.Dense(self.num_outputs)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        # compile
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=params.learning_rate),
                      loss='mse',
                      metrics=[keras.metrics.RootMeanSquaredError()])
        print("returning model")
        return model


    def create_bilstm_hypermodel(self, hp):
        """
            @brief: the hypermodel function passed to the tuner, calls build_model
        """
        # tune the number of layers
        max_layers = 6
        min_layers = 3
        layers = 3#hp.Int('layers', min_value=min_layers, max_value=max_layers, step=1)

        # tune the number of units per layer
        min_units = 32
        max_units = 64
        units = hp.Int('units', min_value=min_units, max_value=max_units, step=32)
        # units = []
        # for i in range(max_layers):
        #     units.append(hp.Int(f'units_{i}', min_value=min_units, max_value=max_units, step=32))

        # tune the dropout rate
        dropout_rate = hp.Choice('dropout_rate', values=[.2, .25, .4, .5])

        # tune regularization rate
        l1 = hp.Choice('l1', values=[.000001, .000005, .00001, .00005, .0001, .0005, .001])
        l2 = -1

        recurrent_dropout = 0#hp.Choice('recurrent_dropout', values = [0.0, .25, .5])
        # tune the learning rate for the optimizer
        learning_rate = hp.Choice('learning_rate', values=[.0001, .00025, .0005, .00075, .001, .0025, .005])

        params = MyParameters(layers=layers, units=units, dropout_rate=dropout_rate, recurrent_dropout=recurrent_dropout, l1=l1, l2=l2, learning_rate=learning_rate)

        return self.build_bilstm_model(params)




    def build_bilstm_model(self, params, *args, **kwargs):
        """
            @brief: builds the model with the specified params
        """
        # input layer
        inputs = keras.Input(shape=self.input_shape, name='in1')
                    # first layer
        x = layers.Bidirectional(layers.LSTM(units=params.units, 
                            recurrent_dropout=params.recurrent_dropout,
                            kernel_regularizer=regularizers.l1_l2(l1=params.l1, l2=params.l2),
                            return_sequences=True, 
                            name=f'hidden_{0}'))(inputs)

        if params.dropout_rate > 0.0:
            x = layers.Dropout(rate=params.dropout_rate)(x)

        # subsequent layers
        for i in range(1, params.layers-1):
            x = layers.Bidirectional(layers.LSTM(units=params.units, 
                            recurrent_dropout=params.recurrent_dropout,
                            kernel_regularizer=regularizers.l1_l2(l1=params.l1, l2=params.l2),
                            return_sequences=True, 
                            name=f'hidden_{i}'))(x)
            
            if params.dropout_rate > 0.0:
                x = layers.Dropout(rate=params.dropout_rate)(x)
        
        # last layers
        x = layers.Bidirectional(layers.LSTM(units=params.units, 
                            recurrent_dropout=params.recurrent_dropout,
                            kernel_regularizer=regularizers.l1_l2(l1=params.l1, l2=params.l2),
                            return_sequences=False, 
                            name=f'hidden_{i+1}'))(x)

        if params.dropout_rate > 0.0:
            x = layers.Dropout(rate=params.dropout_rate)(x)     

        # output layer
        outputs = layers.Dense(self.num_outputs)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        # compile
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=params.learning_rate),
                      loss=self.loss_fcn,#'mse',
                      metrics=[keras.metrics.RootMeanSquaredError()])
        print("returning model")
        return model


    def create_cnn_hypermodel(self, hp):
        pass


    def build_cnn_hypermodel(self, params, *args, **kwargs):
        pass



    def hyperband_search(self,
                         objective: str = 'root_mean_squared_error',
                         mode: str = 'min',
                         max_epochs: int = 10,
                         factor: int = 2,
                         hyperband_iterations: int = 3,
                         executions_per_trial: int = 3,
                         hypermodel: kt.HyperModel = None,
                         directory: str = None,
                         project_name: str = None,
                         logger: TensorBoardLogger = None,
                         X: object = None,
                         y: object = None):

        hyperband = MyTuner(oracle=kt.oracles.HyperbandOracle(
                                objective=kt.Objective(objective, mode),
                                max_epochs=max_epochs,
                                factor=factor,
                                hyperband_iterations=hyperband_iterations
                                 ),
                            executions_per_trial=executions_per_trial,
                            hypermodel=hypermodel,
                            directory=directory,
                            project_name=project_name,
                            logger=logger
        )

        setup_tb(hyperband)
        hyperband.search(X,
                         y,
                         epochs=max_epochs,
                         validation_split=.2)

        return hyperband



    def random_search(self,
                     objective: str = 'root_mean_squared_error',
                     mode: str = 'min',
                     max_trials: int = 256,
                     epochs: int = 10,
                     executions_per_trial: int = 3,
                     hypermodel: kt.HyperModel = None,
                     directory: str = None,
                     project_name: str = None,
                     logger: TensorBoardLogger = None,
                     X: object = None,
                     y: object = None):

        randomsearch = MyTuner(oracle=kt.oracles.RandomSearchOracle(
                                objective=kt.Objective(objective, mode),
                                max_trials=max_trials,
                                 ),
                            executions_per_trial=executions_per_trial,
                            hypermodel=hypermodel,
                            directory=directory,
                            project_name=project_name,
                            logger=logger
        )

        setup_tb(randomsearch)
        randomsearch.search(X,
                         y,
                         epochs=epochs,
                         validation_split=.2)

        return randomsearch



    def bayesian_search(self,
                        objective: str = 'root_mean_squared_error',
                        mode: str = 'min',
                        max_trials: int = 256,
                        epochs: int = 2,
                        alpha: float = .00025,
                        beta: float = 2.75,
                        executions_per_trial: int = 3,
                        hypermodel: kt.HyperModel = None,
                        directory: str = None,
                        project_name: str = None,
                        logger: TensorBoardLogger = None,
                        X: object = None,
                        y: object = None):

        bayesian = MyTuner(oracle=kt.oracles.BayesianOptimizationOracle(
                            objective=kt.Objective(objective, mode),
                            max_trials=max_trials,
                            alpha=alpha,
                            beta=beta
                        ),
                        executions_per_trial=executions_per_trial,
                        hypermodel=hypermodel,
                        directory=directory,
                        project_name=project_name,
                        logger=logger
        )

        setup_tb(bayesian)
        bayesian.search(X,
                         y,
                         epochs=epochs,
                         validation_split=.2)

        return bayesian


###############################################################################################################
###############################################################################################################

class MyTuner(kt.Tuner):
    """
        @brief: overrides built-in

        @functions:
            run_trial: allows for batch_size to be tuned
    """
    def run_trial(self, trial, *args, **kwargs):
        min_batch_size = 256 if not 'min_batch_size' in kwargs else kwargs['min_batch_size']
        max_batch_size = 512 if not 'max_batch_size' in kwargs else kwargs['max_batch_size']
        batch_size_step = 256 if not 'batch_size_step' in kwargs else kwargs['max_batch_size']

        kwargs['batch_size'] = 256#trial.hyperparameters.Int('batch_size', min_batch_size, max_batch_size, step=batch_size_step)
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
        self.l1 = kwargs['l1']
        self.learning_rate = kwargs['learning_rate']
        x = 1 if not 'a' in kwargs else kwargs['a']
        self.metric = "na" if not 'metric' in kwargs else kwargs['metric']
        self.score = -1 if not 'score' in kwargs else kwargs['score']
        self.recurrent_dropout = kwargs['recurrent_dropout']

    def __str__(self):
        return f"{self.layers}-layers_{self.units}-units_{str(self.learning_rate).split('.')[1]}_learningRate"

    def __repr__(self):
        return f"{self.layers}-layers_{self.units}-units_{str(self.learning_rate).split('.')[1]}_learningRate"















