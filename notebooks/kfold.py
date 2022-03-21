

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



params = {'datasource.username': 'macslab', # the username of the logged in user
            'datasource.password': 'Ch0colate!', 
            'datasource.database': 'ncmapss_db', # <- NO CHANGE 
            'datasource.url': '10.2.219.98', # <- or your database installation location
            'datasource.port': '5432'} # <- most likely don't change
db, cur =  api.connect(params)
db.set_session(autocommit=True)
del(params)


units_df = api._get_units(db=db)
units = units_df[(units_df['Fc'] == Fc) & (units_df['dataset'].str.contains(dataset))]


tables = ['summary_tb', 'telemetry_tb']
downsample=20
df = api._get_data(db=db,
                   units=pd.unique(units.id),
                   tables=tables,
                   downsample=downsample).astype(np.float32)
utils.add_time_column(units=pd.unique(units.id), df=df)
utils.add_rul_column(units=pd.unique(units.id), df=df)


W_cols = ['Mach', 'alt', 'TRA', 'T2', 'time']
Xs_cols = ['Wf', 'Nf', 'Nc', 'T24', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50']
aux_cols = ['cycle', 'hs', 'Fc', 'asset_id']

model = keras.models.load_model(paths_df[paths_df['name']=='flight_effects'].path.values[0])
yscaler = joblib.load(paths_df[paths_df['name']=='flight_effects_yscaler'].path.values[0])
xscaler = joblib.load(paths_df[paths_df['name']=='flight_effects_xscaler'].path.values[0])


trace = yscaler.transform(df[Xs_cols])
pred = model.predict(xscaler.transform(df[W_cols]))
res = trace - pred
dfx = pd.DataFrame(data=res, columns=Xs_cols)
df_x = pd.DataFrame(data=xscaler.transform(df[W_cols]), columns=W_cols)
dfx = pd.concat([dfx, df_x, df[aux_cols]], axis=1)
dfx['rul'] = df['rul'].values
dfx.time = dfx.time + (dfx.cycle -1)
dfx0 = dfx[dfx.hs == 0]



lookback = 100
horizon = 1
n_out = 1
n_features = 19
input_shape = (lookback, n_features)

monitor = 'val_root_mean_squared_error'
mode = 'min'
min_delta = .1
patience = 20

batch_size = 256
epochs = 100

scores = []

traces = []

preds = []

test_units = []

params = []

params.append(tuning.MyParameters(layers=3, units=24, dropout_rate=.2, learning_rate = .00075, recurrent_dropout=0.0, l2=-1, l1=.00001))
params.append(tuning.MyParameters(layers=3, units=32, dropout_rate=.2, learning_rate = .00075, recurrent_dropout=0.0, l2=-1, l1=.00001))
params.append(tuning.MyParameters(layers=3, units=64, dropout_rate=.2, learning_rate = .00075, recurrent_dropout=0.0, l2=-1, l1=.00001))


def decay_schedule(epoch, lr):
    if epoch > 1:
        lr = lr * .99
    return lr

lr_scheduler = keras.callbacks.LearningRateScheduler(decay_schedule)

early_stopping = keras.callbacks.EarlyStopping(monitor=monitor,
                                               mode=mode,
                                               min_delta=min_delta,
                                               patience=patience,
                                               verbose=1,
                                               restore_best_weights=True)#True)

for j in range(len(params)):
    results = {}
    results[f'model_{j}'] = {}
    results[f'model_{j}']['params'] = params[j].__dict__
    
    for i in range(len(units)):
        
        tensorboard = keras.callbacks.TensorBoard(log_dir=f'{log_location}/{data_header}/kmeans/model_{j}/test_unit_{units.iloc[i].id}/',
                                              histogram_freq=1,
                                              write_images=True,
                                              write_graph=True)
        
        test_df = dfx0[dfx0.asset_id == units.iloc[i].id]
        test_y = test_df.pop('rul')
        train_df = dfx0[dfx0.asset_id != units.iloc[i].id]
        train_y = train_df.pop('rul')

        print(f'training on <{[int(x) for x in pd.unique(train_df.asset_id)]}>, testing on <{units.iloc[i].id}>')

        X_train, y_train = utils.temporalize_data(train_df[W_cols + Xs_cols].values, train_y.values, lookback, horizon, n_features, n_out)
        X_test, y_test = utils.temporalize_data(test_df[W_cols + Xs_cols].values, test_y.values, lookback, horizon, n_features, n_out)

        X_train = np.array(X_train).astype(np.float32)
        y_train = np.array(y_train).astype(np.float32)

        X_test = np.array(X_test).astype(np.float32)
        y_test = np.array(y_test).astype(np.float32)

        my_tuning = tuning.Tuning(input_shape, n_out)
                               
        model = my_tuning.build_bilstm_model(params[j])
                                              
        history = model.fit(X_train,
                            y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            shuffle=False,
                            validation_split=.1,
                            callbacks=[early_stopping, lr_scheduler, tensorboard],
                            verbose=1)

        test_score = model.evaluate(X_test, y_test, batch_size=batch_size)

        res = model.predict(X_test)

        scores.append((history.history['val_root_mean_squared_error'][-1] + test_score[1])/2)
        traces.append(y_test.flatten())
        preds.append(res)
        test_units.append(units.iloc[i].id)

        variables = {"val_rmse": history.history['val_root_mean_squared_error'][-1], 
                     "test_rmse": test_score[1], 
                     "test_unit": test_units[i], 
                     "trace": list(traces[i]),
                     "pred": list(preds[i])
                    }

        results[f'model_{j}'][f'data_{i}'] = variables

    with open('kfold_data_{j}.json', 'w') as outfile:
        json.dump(results, outfile)