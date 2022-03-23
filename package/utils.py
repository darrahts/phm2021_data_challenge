import h5py
import numpy as np
import pandas as pd
from pandas import DataFrame
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import boto3
import base64
from botocore.exceptions import ClientError
import json
import os
import string
from collections import Mapping, Container
from sys import getsizeof
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance

h5_dir = 'data_h5/'
csv_dir = 'data_csv/'
exp_dir = 'experiments/'
fnames = ['N-CMAPSS_DS08a-009.h5', 'N-CMAPSS_DS08c-008.h5']
sets = ['dev', 'test']



def check_gpu():
    print(tf.__version__)
    gpu = tf.config.list_physical_devices('GPU')
    has_gpu = gpu[0][0].split(':')[1] == 'GPU'
    print(f"[INFO] GPU?: <{has_gpu}> {gpu}")
    if (has_gpu):
        for i in range(len(gpu)):
            print("expanding memory growth")
            tf.config.experimental.set_memory_growth(gpu[i], True)
    return has_gpu



def load_h5(fnames: list = [],
            sets: list = [],
            h5_dir: str = '',
            csv_dir: str = '',
            save: int = 1,  # 0 = dont save, 1 = save, 2 = save and return
            save_as: str = '',
            verbose: bool = False) -> tuple:
    """
    basically the same code provided in the template but saves data to files
    prefer to use the file populate_db, which is uses the database
    """
    df = pd.DataFrame()
    asset_id = 1

    for filename in fnames:
        if verbose:
            print(filename)
        for _set in sets:
            if verbose:
                print(_set)
            with h5py.File(h5_dir + filename, 'r') as hdf:
                a_data = np.array(hdf.get(f"A_{_set}"))
                w_data = np.array(hdf.get(f"W_{_set}"))
                x_data = np.array(hdf.get(f"X_s_{_set}"))
                v_data = np.array(hdf.get(f"X_v_{_set}"))
                t_data = np.array(hdf.get(f"T_{_set}"))
                y_data = np.array(hdf.get(f"Y_{_set}"))

                a_labels = [l.decode('utf-8') for l in list(np.array(hdf.get('A_var')))]
                w_labels = [l.decode('utf-8') for l in list(np.array(hdf.get('W_var')))]
                x_labels = [l.decode('utf-8') for l in list(np.array(hdf.get('X_s_var')))]
                v_labels = [l.decode('utf-8') for l in list(np.array(hdf.get('X_v_var')))]
                t_labels = [l.decode('utf-8') for l in list(np.array(hdf.get('T_var')))]

            df_a = DataFrame(data=a_data, columns=a_labels)
            df_a['asset_id'] = -1
            df_a['dataset'] = filename.split('_')[1].split('.')[0]
            df_w = DataFrame(data=w_data, columns=w_labels)
            df_x = DataFrame(data=x_data, columns=x_labels)
            df_v = DataFrame(data=v_data, columns=v_labels)
            df_t = DataFrame(data=t_data, columns=t_labels)
            df_y = DataFrame(data=y_data, columns=['y'])
            if verbose:
                print(f"<{filename}> : {pd.unique(df_a.unit)}")
            for n in list(pd.unique(df_a.unit)):
                df_a.loc[df_a['unit'] == n, 'asset_id'] = asset_id
                asset_id = asset_id + 1

            df_temp = pd.concat([df_a, df_y, df_w, df_x, df_v, df_t], axis=1)
            if verbose:
                print(df_temp.head())
            if (len(df)) == 0:
                df = df_temp
            else:
                df = pd.concat([df, df_temp], axis=0)

    df.asset_id = df.asset_id.astype(int)
    df.unit = df.unit.astype(int)
    df.cycle = df.cycle.astype(int)
    df.hs = df.hs.astype(int)
    df.Fc = df.Fc.astype(int)

    df_aux = df[['asset_id', 'Fc', 'unit', 'dataset', 'cycle']].groupby('asset_id').agg({'Fc': 'max',
                                                                             'unit': 'max',
                                                                             'dataset': 'max',
                                                                             'cycle': ['min', 'max']})
    df_aux.reset_index(inplace=True)
    df_aux.columns = ['asset_id', 'group_id', 'unit', 'dataset', 'age', 'eol']
    df_aux.age = df_aux.age - 1.0
    df_aux.drop(columns=['asset_id'], inplace=True)

    y_labels = t_labels
    t_labels = [w_labels, x_labels]
    t_labels = [l for labels in t_labels for l in labels]
    if verbose:
        print(y_labels)
        print(t_labels)
        
    if save > 0:
        if verbose:
            print(f"[INFO] saving dataframe.")
        df.to_csv(csv_dir + save_as)
        if verbose:
            print(f"[INFO] saving aux dataframe.")
        df_aux.to_csv(csv_dir + 'aux_' + save_as)

        with open(csv_dir + 'y_labels.txt', "w") as f:
            for l in y_labels:
                f.write(f"{l}\n")

        with open(csv_dir + 't_labels.txt', "w") as f:
            for l in t_labels:
                f.write(f"{l}\n")

        with open(csv_dir + 'v_labels.txt', "w") as f:
            for l in v_labels:
                f.write(f"{l}\n")

    if save == 2 or save == 0:
        return df, df_aux, y_labels, t_labels, v_labels



def resample_df(df: pd.DataFrame = pd.DataFrame(),
                factor: int = 10,
                interp_method: str = 'time',
                csv_dir: str = '',
                save: int = 1,
                save_as: str = '',
                drop_column_zero: bool = True) -> pd.DataFrame:
    if drop_column_zero:
        df.drop(columns=[df.columns[0]], inplace=True)
    df.index = pd.to_timedelta(df.index, unit='s')
    df = df.resample(f'{factor}S').interpolate(method=interp_method)
    if save > 0:
        df.to_csv(csv_dir + save_as)
    if save == 2 or save == 0:
        return df



def interp_y(df: pd.DataFrame = None,
             csv_dir: str = '',
             save: int = 1,
             save_as: str = '') -> pd.DataFrame:
    units = pd.unique(df.asset_id)
    for u in range(0, len(units)):
        cycles = pd.unique(df[df.asset_id == u].cycle)
        for c in range(1, len(cycles)):
            y1 = df[(df.asset_id == u) & (df.cycle == c)].y.values[0]
            y0 = y1 - 1
            n = len(df[(df.asset_id == u) & (df.cycle == c)])
            yx = np.linspace(y1, y0, n)
            df.loc[((df.asset_id == u) & (df.cycle == c)), 'y'] = yx
    if save > 0:
        df.to_csv(csv_dir + save_as)
    if save == 2 or save == 0:
        return df



def train_test_split_old(df: pd.DataFrame = None,
                     train_pct: float = .65,
                     val_pct: float = .2,
                     test_pct: float = .15,
                     verbose: bool = True) -> tuple:
    """
    DEPRACATED, TO BE REMOVED
    """
    asset_id = list(pd.unique(df.asset_id))
    samples = len(asset_id)
    if verbose:
        print(f"unit unique identifier (asset_id): {asset_id}")
        print(f"number of units: {samples}")

    train_cnt = int(samples * train_pct)
    val_cnt = int(samples * val_pct)
    test_cnt = int(samples * test_pct) + 1
    if verbose:
        print(f"train, val, test set counts: {train_cnt}, {val_cnt}, {test_cnt}")

    assert train_cnt + val_cnt + test_cnt == samples, "error"

    train_asset_id = random.sample(asset_id, train_cnt)
    asset_id = list(set(asset_id) - set(train_asset_id))
    val_asset_id = random.sample(asset_id, val_cnt)
    asset_id = list(set(asset_id) - set(val_asset_id))
    test_asset_id = random.sample(asset_id, test_cnt)
    asset_id = list(set(asset_id) - set(test_asset_id))

    assert len(asset_id) == 0, "error"

    if verbose:
        print(f"train asset_id: {train_asset_id}")
        print(f"val asset_id: {val_asset_id}")
        print(f"test asset_id: {test_asset_id}")

    train_df = df[df['asset_id'].isin(train_asset_id)]
    val_df = df[df['asset_id'].isin(val_asset_id)]
    test_df = df[df['asset_id'].isin(test_asset_id)]

    train_y = np.array(train_df.y, dtype=np.float32)
    val_y = np.array(val_df.y, dtype=np.float32)
    test_y = np.array(test_df.y, dtype=np.float32)

    return train_df, train_y, val_df, val_y, test_df, test_y


def train_test_split(df: pd.DataFrame = None,
                    units: [] = None,
                     y_labels: [] = None,
                     t_labels: [] = None,
                     train_pct: float = .75,
                     val_pct: float = .10,
                     test_pct: float = .15,
                     verbose: bool = False):
    # if units.any():
    #     units = units
    #     print(units)
    #     print(type(units))
    # else:
    #     print('here')
    #     units = list(pd.unique(df.asset_id))
    num_units = len(units)
    print(num_units)
    train_cnt = int(num_units * train_pct)
    print(train_cnt)
    val_cnt = int(num_units * val_pct)
    print(val_cnt)
    test_cnt = int(num_units * test_pct)
    print(test_cnt)


    if verbose:
        print(f"train, val, test set counts: {train_cnt}, {val_cnt}, {test_cnt}")

    print(num_units, (train_cnt+val_cnt+test_cnt))
    if num_units - (train_cnt + val_cnt + test_cnt) == 2:
        #train_cnt += 1
        val_cnt += 1
        test_cnt += 1
        
    assert train_cnt + val_cnt + test_cnt == num_units, "error1"

    train_units = random.sample(units, train_cnt)
    units = list(set(units) - set(train_units))

    val_units = random.sample(units, val_cnt)
    units = list(set(units) - set(val_units))

    test_units = random.sample(units, test_cnt)
    units = list(set(units) - set(test_units))

    assert len(units) == 0, "error2"

    if verbose:
        print(f"train units: {train_units}")
        print(f"val units: {val_units}")
        print(f"test units: {test_units}")

    train_df = df[df['asset_id'].isin(train_units)]
    val_df = df[df['asset_id'].isin(val_units)]
    test_df = df[df['asset_id'].isin(test_units)]

    train_y = np.array(train_df[y_labels], dtype=np.float32)
    val_y = np.array(val_df[y_labels], dtype=np.float32)
    test_y = np.array(test_df[y_labels], dtype=np.float32)
    
    train_df = train_df[t_labels]
    val_df = val_df[t_labels]
    test_df = test_df[t_labels]
    
    return train_df, train_y, val_df, val_y, test_df, test_y


def temporalize_data(inputs, outputs, lookback, horizon, n_features, n_out):
    """
        @brief: reshapes a dataset to account for time

        @params:
            inputs: the 'X' vector (what you are training on)
            outputs: the 'y' vector (what you are predicting)
            lookback: the number of samples in the lookback period (NOTE: not
                        the same as lookback_hours)
            horizon: the number of samples in the horizon period (NOTE: not
                        the same as horizon_hours)
            n_features: the number of features in inputs (X)
            n_out: the number of outputs in outputs (y)

        @returns:
            X_out: a 3d numpy matrix of size [sz, lookback, n_features]
            y: a 3d numpy matrix of size [sz, horizon, n_out]
    """
    X_out = []
    y = []
    ################# the number of samples to generate accounts for the lookback and horizon size
    sz = (len(inputs) - (lookback + horizon) - 1)
    for i in range(0, sz):
        temp = []
        for j in range(0, lookback):
            ################# temp stacks sequential observations and creates a single sample
            temp.append(inputs[[i + j]])
        ################# X_out[0] is now a single sample array with the first [0 - <lookback>] observations
        X_out.append(temp)
        temp = []
        if (len(outputs) > 0):
            ################# the subsequent observations are the values to predict
            for j in range(lookback, lookback + horizon):
                temp.append(outputs[[i + j]])
            ################# y[0] is now a single sample array with the first [<lookback> - <horizon>] observations
            y.append(temp)
    ################# convert to numpy arrays and reshape for LSTM input
    X_out = np.array(X_out)
    X_out = X_out.reshape(X_out.shape[0], lookback, n_features)
    y = np.array(y)
    y = y.reshape(y.shape[0], horizon, n_out)
    return X_out, y



def make_plot(type: str = '',
              save_fig: int = 0,
              X_val: np.array = None,
              y_val: np.array = None,
              y_test: np.array = None,
              y_pred: np.array = None,
              history: dict = {},
              directory: str = '',
              input_name: str = ''): # 0 is no and it will only display, 1 is yes without display, 2 is yes with display

    if 'rul_plot' in type:
        fig = plt.figure(figsize=(9, 6))
        plt.plot(y_test.reshape(-1, 1), label='true')
        if 'dnn' in directory:
            plt.plot(y_pred[:, 1, 0], label='pred')
        else:
            plt.plot(y_pred, label='pred')
        plt.title("test results")
        plt.xlabel("sample number")
        plt.ylabel(f"rul for {input_name} at {directory}")
        plt.legend()

    elif 'error_plot' in type:
        fig = plt.figure(figsize=(9, 6))
        plt.plot(history['root_mean_squared_error'], label='training')
        if X_val is not None and y_val is not None:
            plt.plot(history['val_root_mean_squared_error'], label='validation')
        plt.legend()
        plt.title(f"rmse for {input_name} at {directory}")
        fig.savefig(directory + 'rmse.png')

    elif 'loss_plot' in type:
        fig = plt.figure(figsize=(9, 6))
        plt.plot(history['loss'], label='training')
        if X_val is not None and y_val is not None:
            plt.plot(history['val_loss'], label='validation')
        plt.legend()
        plt.title(f"loss for {input_name} at {directory}")

    else:
        print("[ERROR] valid type not specified. returning.")
        return

    if save_fig == 0:
        fig.show()
    if save_fig > 0:
        if 'rul' in type:
            fig.savefig(directory + 'results.png')
        elif 'error_plot' in type:
            fig.savefig(directory + 'rmse.png')
        elif 'loss_plot' in type:
            fig.savefig(directory + 'loss.png')
    if save_fig > 1:
        fig.show()



def get_aws_secret(secret_name: str = "", region_name: str = "us-east-1") -> {}:
        """
            @brief: retrieves a secret stored in AWS Secrets Manager. Reqasset_idres AWS CLI and IAM user profile properly configured.

            @input:
                secret_name: the name of the secret
                region_name: region of use, default=us-east-1

            @output:
                secret: dictionary
        """
        client = boto3.session.Session().client(service_name='secretsmanager', region_name=region_name)
        secret = '{"None": "None"}'
        if (len(secret_name) < 1):
            print("[ERROR] no secret name provided.")
        else:
            try:
                res = client.get_secret_value(SecretId=secret_name)
                if 'SecretString' in res:
                    secret = res['SecretString']
                elif 'SecretBinary' in res:
                    secret = base64.b64decode(res['SecretBinary'])
                else:
                    print("[ERROR] secret keys not found in response.")
            except ClientError as e:
                print(e)

        return json.loads(secret)



def get_current_user():
    user = os.environ.get('USER')
    if user is None:
        user = os.environ.get('USERNAME')
    if user is None:
        user = 'user'
    return user



def generate_serial_number(length: int = 8) -> str:
    return ''.join(random.choices(string.digits + string.ascii_letters, k=length))


def load_json(log_location, data_header, footer):
    fname = log_location + '/' + data_header + footer
    with open(fname, 'r') as f:
        data = json.loads(f.read())
    return data


def parse_json(json_object, target_key, res=[]):
    if type(json_object) is dict and json_object:
        for key in json_object:
            if key == target_key:
                res.append(json_object[key])
            parse_json(json_object[key], target_key, res)

    elif type(json_object) is list and json_object:
        for item in json_object:
            parse_json(item, target_key, res)
    
    return res


def chunk_generator(X, n):
    """
    breaks large data into smaller equal pieces + remainder as last yield
    """
    for i in range(0, len(X), n):
        yield i+1, X[i:i+n]



def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    #plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_rmse(history):
    plt.plot(history.history['root_mean_squared_error'], label='root_mean_squared_error')
    plt.plot(history.history['val_root_mean_squared_error'], label='val_root_mean_squared_error')
    #plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_test_unit():
    pass


def plot_feature_distributions(df: pd.DataFrame = None,
                               feature_range: tuple = (-1,1),
                               figsize: tuple = (12,4)) -> None:
    scaler = MinMaxScaler(feature_range=feature_range)
    _df = df.copy()
    _df = pd.DataFrame(data=scaler.fit_transform(_df), columns=df.columns)
    _df.head()

    _plt = _df.melt(var_name='Feature', value_name='Normalized')
    plt.figure(figsize=(figsize))
    ax = sns.violinplot(x='Feature', y='Normalized', data=_plt)
    _ = ax.set_xticklabels(_df.keys(), rotation=45)
    plt.title("Normalized Feature Distribution")
    plt.show()


def plot_feature_importance(features: np.ndarray = None,
                            feature_labels: [] = None,
                            target: np.ndarray = None,
                            target_label: [] = None,
                            figsize: tuple = (9,4)) -> None:
        importance_model = XGBRegressor()
        importance_model.fit(features, target)
        importance = importance_model.feature_importances_

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        plt.bar([x for x in range(len(importance))], importance)
        if feature_labels is not None:
            ax.set_xticks([x for x in range(len(feature_labels))])
            ax.set_xticklabels(feature_labels, rotation=45)
        plt.title(f'Feature importance for target <{target_label}>')
        plt.show()

def plot_feature_importances(model: object = None,
                            features: np.ndarray = None,
                            feature_labels: [] = None,
                            target: np.ndarray = None,
                            target_label: [] = None,
                            scoring: [] = None) -> None:

        results = permutation_importance(estimator=model,
                                         X=features,
                                         y=target,
                                         scoring=scoring)

        importances = [results[x].importances_mean for x in scoring]

        height = 3 * len(scoring)

        fig = plt.figure(figsize=(9, height))
        plt.title(f'Feature importance for target <{target_label}>')

        for i in range(0, len(scoring) - 1):
            ax = fig.add_subplot(len(scoring), 1, i + 1)
            plt.bar([x for x in range(len(importances[i]))], importances[i])
            plt.ylabel(scoring[i])
            ax.set_xticks([])

        ax = fig.add_subplot(len(scoring), 1, i + 2)
        plt.bar([x for x in range(len(importances[i + 1]))], importances[i + 1])
        if feature_labels is not None:
            ax.set_xticks([x for x in range(len(feature_labels))])
            ax.set_xticklabels(feature_labels, rotation=45)
            plt.ylabel(scoring[i + 1])

        plt.show()

def plot_scatter_results(results_df,
                         std_df,
                         x,
                         y):
    results_df = results_df.sort_values(by=[y])
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=results_df, s=250, x=x, y=y, hue='sizes', palette='Dark2')  # , dodge=False)
    sns.scatterplot(x=results_df[x], s=150, y=results_df[y] + std_df[y], marker='^', hue=results_df['sizes'],
                    palette='Dark2')
    sns.scatterplot(x=results_df[x], s=150, y=results_df[y] - std_df[y], marker='v', hue=results_df['sizes'],
                    palette='Dark2')
    for i in range(len(results_df)):
        plt.plot([results_df[x][i], results_df[x][i]],
                 [results_df[y][i] - std_df[y][i], results_df[y][i] + std_df[y][i]], color='gray')
    plt.legend('')
    plt.title(f"Results Scatter Plot ({x} x {y})")
    plt.show()



def load_model(model_location, data_header, model_name, model_number):
    return tf.keras.models.load_model(f'{model_location}/{data_header}/{model_name}/model_{model_number}')


def plot_rul_results(files, data_header, notes=''):
    for i in range(len(files)):
        params = parse_json(files[i][f'model_{i}'], 'params', [])[0]
        num_samples = len([x for x in files[i][f'model_{i}'].keys() if 'data' in x])
        
        for j in range(num_samples):
            test_unit = parse_json(files[i][f'model_{i}'][f'data_{j}'], 'test_unit', [])[0]
            val_unit = parse_json(files[i][f'model_{i}'][f'data_{j}'], 'val_unit', [])[0]
            test_rmse = round(parse_json(files[i][f'model_{i}'][f'data_{j}'], 'test_rmse', [])[0], 3)
            val_rmse = round(parse_json(files[i][f'model_{i}'][f'data_{j}'], 'val_rmse', [])[0], 3)
            trace = parse_json(files[i][f'model_{i}'][f'data_{j}'], 'trace', [])[0]
            pred = parse_json(files[i][f'model_{i}'][f'data_{j}'], 'pred', [])[0]


            fig = plt.figure(figsize=(5,5))
            plt.plot(np.array(trace), c='b', label='actual')
            plt.plot(np.array(pred), c='r', label='prediction')
            plt.legend()
            plt.xlabel('sample')
            plt.ylabel('RUL (cycles)')
            plt.text(5000, 10, f'test rmse: {test_rmse}')
            plt.text(5000, 7, f'val rmse: {val_rmse}')
            plt.text(5000, 2, data_header)
            if notes:
                plt.text(500, 1, notes)
            plt.title(f"test unit: {test_unit}, model: {params['layers']}-layers_{params['units']}-units_{str(params['dropout_rate']).replace('.', '')}-dropout")
            plt.show()


def transform():
    pass


def add_time_column(units, df):
    df['time'] = ""
    for unit in units:
        for cycle in pd.unique(df[df.asset_id == unit].cycle):
            length = len(df[(df.asset_id == unit) & (df.cycle == cycle)])
            x = np.arange(0, length)/(length-1)
            df.loc[(df.asset_id == unit) & (df.cycle == cycle), 'time'] = x
    #df.time = df.time + (df.cycle -1)


def add_rul_column(units, df):
    df['rul'] = ""
    for unit in units:
        eol = max(df[df.asset_id == unit].cycle)
        for cycle in pd.unique(df[df.asset_id == unit].cycle):
            length = len(df[(df.asset_id == unit) & (df.cycle == cycle)])
            x = np.arange(0, length)/(length-1)
            df.loc[(df.asset_id == unit) & (df.cycle == cycle), 'rul'] = eol - (cycle-1) - x


def outlier_removal_z_score(data):
    mean = np.zeros((data.shape[1],), dtype=np.float32)
    std = np.zeros((data.shape[1],), dtype=np.float32)

    for i in range(data.shape[1]):
        mean[i] = np.mean(data[:,i])
        std[i] = np.std(data[:,i])

    for i in range(data.shape[0]):
        z = (data[i,:] - mean) / std
        outliers = [i for i, x in enumerate((z > 3) | (z < -3)) if x]
        if outliers:
            data[i,outliers] = mean[outliers]

    return data



def outlier_removal(df):
    pass