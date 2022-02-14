#!/usr/bin/env python
# coding: utf-8

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


# THESE ARE YOUR CREDENTIALS IN PLAIN TEXT!
params = {'datasource.username': 'macslab', # the username of the logged in user
            'datasource.password': 'Ch0colate!', 
            'datasource.database': 'ncmapss_db', # <- NO CHANGE 
            'datasource.url': 'localhost', # <- or your database installation location
            'datasource.port': '5432'} # <- most likely don't change
#print(params)
db, cur =  api.connect(params)
db.set_session(autocommit=True)
del(params)




units = api._get_units(db=db)
units.head()


tables = ['summary_tb', 'telemetry_tb']
downsample=20
df = api._get_data(db=db,
                   units=list(np.arange(1,40,1)),#pd.unique(units.id),
                   tables=tables,
                   downsample=downsample).astype(np.float32)
utils.add_time_column(units=list(np.arange(1,40,1)), df=df)
utils.add_rul_column(units=list(np.arange(1,40,1)), df=df)


X_cols = ['Mach', 'alt', 'TRA', 'T2', 'time']## disregard comments #, 'Fc'] # will also have Fc_1, Fc_2, Fc_3
y_cols = ['Wf', 'Nf', 'Nc', 'T24', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50']
a_cols = ['cycle', 'hs', 'Fc', 'asset_id']

model = keras.models.load_model('models/ncmapss/flight_effects.h5')
yscaler = joblib.load('models/ncmapss/flight_effects_yscaler.pkl')
xscaler = joblib.load('models/ncmapss/flight_effects_xscaler.pkl')

Xs_prime = []
idx = np.random.randint(len(y_cols))
polys = []

for asset_id in pd.unique(df.asset_id):
    X_full = df[df.asset_id == asset_id][X_cols]
    pred = model.predict(xscaler.transform(X_full))
    Xs_prime.append(pred)


df[y_cols] = yscaler.transform(df[y_cols])
df.head()


len(list(np.arange(1,40,1)))


train_df, train_y, val_df, val_y, test_df, test_y = utils.train_test_split(df=df, units=list(np.arange(1,40,1)), y_labels=['rul'], t_labels=a_cols + y_cols, train_pct = .40, val_pct=.30, test_pct=.30, verbose=True)

train_pop = pd.concat([train_df.pop(x) for x in a_cols], axis=1)
val_pop = pd.concat([val_df.pop(x) for x in a_cols], axis=1)
test_pop = pd.concat([test_df.pop(x) for x in a_cols], axis=1)


important_features = ['Wf','Nc', 'T24', 'T48', 'T50', 'P2', 'Ps30', 'P40']


lookback = 100
horizon = 1
n_out = 1
n_features = train_df[important_features].shape[1]

X_train, y_train = utils.temporalize_data(train_df[important_features].values, train_y, lookback, horizon, n_features, n_out)

input_shape = (lookback, n_features)
my_tuning = tuning.Tuning(input_shape, n_out)
bayesian_tuning = my_tuning.bayesian_search(objective='root_mean_squared_error',
                                            mode='min',
                                            max_trials=256,
                                            alpha=.00025,
                                            beta=2.75,
                                            epochs=3,
                                            executions_per_trial=1,
                                            hypermodel=my_tuning.create_lstm_hypermodel,
                                            directory='logs/ncmapss/bayesiansearch',
                                            project_name='lstm',
                                            logger=TensorBoardLogger(
                                                metrics=['root_mean_squared_error'],
                                                         logdir='logs/ncmapss/hparams'
                                            ),
                                            X=X_train,
                                            y=y_train)

bayesian_tuning_params = bayesian_tuning.get_best_hyperparameters(num_trials=1)[0]
bayesian_tuning_model = bayesian_tuning.get_best_models()[0]
print(bayesian_tuning_params.values)
bayesian_tuning_model.summary()


save_dir = 'models/ncmapss/'
name = 'rul_best_model'

bayesian_tuning_model.save(save_dir + name + '.h5')
bayesian_tuning_model.save(save_dir + name)


