import os
import sys
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

base_dir = os.path.dirname(os.getcwd())
print(base_dir)
sys.path.insert(1, base_dir)
from package.api import DB as api
import package.utils as utils
import package.tuning as tuning
utils.check_gpu()

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, optimizers, metrics
#import tensorflow_addons as tfa

import keras_tuner as kt

from kerastuner_tensorboard_logger import (
    TensorBoardLogger,
    setup_tb  # Optional
)

paths_df = pd.read_csv(base_dir + '/paths.csv')
paths_df['path'] = base_dir + '/' + paths_df['path']

Fc = 3
dataset = 'DS08'

log_location = base_dir + '/logs'
model_location = base_dir + '/models'
data_location = base_dir + '/data'
data_header = f'Fc-{Fc}_dataset-{dataset}'


X_train = np.load(f'{data_location}/{data_header}_X.npy').astype(np.float32)
y_train = np.load(f'{data_location}/{data_header}_y.npy').astype(np.float32)

input_shape = (lookback, n_features)
my_tuning = tuning.Tuning(input_shape, n_out)
bayesian_tuning = my_tuning.bayesian_search(objective='root_mean_squared_error',
                                            mode='min',
                                            max_trials=128,
                                            alpha=.0002,
                                            beta=4,
                                            epochs=5,
                                            executions_per_trial=1,
                                            hypermodel=my_tuning.create_bilstm_hypermodel,
                                            directory=f'{log_location}/{data_header}',
                                            project_name='lstm',
                                            logger=TensorBoardLogger(
                                                metrics=['root_mean_squared_error'],
                                                         logdir=f'{log_location}/{data_header}/hparams'
                                            ),
                                            X=X_train,
                                            y=y_train)

bayesian_tuning_params = bayesian_tuning.get_best_hyperparameters(num_trials=1)[0]
bayesian_tuning_model = bayesian_tuning.get_best_models()[0]
print(bayesian_tuning_params.values)
bayesian_tuning_model.summary()


bayesian_tuning_model.save(f'{model_location}/{data_header}_best.h5')
bayesian_tuning_model.save(f'{model_location}/{data_header}_best')
