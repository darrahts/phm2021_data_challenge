import h5py
import numpy as np
import pandas as pd
from pandas import DataFrame
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import boto3
import base64
from botocore.exceptions import ClientError
import json
import os
import string

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
    if(has_gpu):
        tf.config.experimental.set_memory_growth(gpu[0], True)
    return has_gpu



def load_h5(fnames: list = [],
            sets: list = [],
            h5_dir: str = '',
            csv_dir: str = '',
            save: int = 1,  # 0 = dont save, 1 = save, 2 = save and return
            save_as: str = '',
            verbose: bool = False) -> tuple:

    df = pd.DataFrame()
    ui = 0

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
            df_a['ui'] = -1
            df_a['dataset'] = filename.split('_')[1].split('.')[0]
            df_w = DataFrame(data=w_data, columns=w_labels)
            df_x = DataFrame(data=x_data, columns=x_labels)
            df_v = DataFrame(data=v_data, columns=v_labels)
            df_t = DataFrame(data=t_data, columns=t_labels)
            df_y = DataFrame(data=y_data, columns=['y'])
            if verbose:
                print(f"<{filename}> : {pd.unique(df_a.unit)}")
            for n in list(pd.unique(df_a.unit)):
                df_a.loc[df_a['unit'] == n, 'ui'] = ui
                ui = ui + 1

            df_temp = pd.concat([df_a, df_y, df_w, df_x, df_v, df_t], axis=1)
            if verbose:
                print(df_temp.head())
            if (len(df)) == 0:
                df = df_temp
            else:
                df = pd.concat([df, df_temp], axis=0)

    df_aux = df[['ui', 'Fc', 'unit', 'dataset', 'cycle']].groupby('ui').agg({'Fc': 'max',
                                                                             'unit': 'max',
                                                                             'dataset': 'max',
                                                                             'cycle': ['min', 'max']})
    df_aux.reset_index(inplace=True)
    df_aux.columns = ['ui', 'group_id', 'unit', 'dataset', 'age', 'eol']
    df_aux.age = df_aux.age - 1.0
    df_aux.drop(columns=['ui'], inplace=True)

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
    units = pd.unique(df.ui)
    for u in range(0, len(units)):
        cycles = pd.unique(df[df.ui == u].cycle)
        for c in range(1, len(cycles)):
            y1 = df[(df.ui == u) & (df.cycle == c)].y.values[0]
            y0 = y1 - 1
            n = len(df[(df.ui == u) & (df.cycle == c)])
            yx = np.linspace(y1, y0, n)
            df.loc[((df.ui == u) & (df.cycle == c)), 'y'] = yx
    if save > 0:
        df.to_csv(csv_dir + save_as)
    if save == 2 or save == 0:
        return df



def train_test_split(df: pd.DataFrame = None,
                     train_pct: float = .65,
                     val_pct: float = .2,
                     test_pct: float = .15,
                     verbose: bool = True) -> tuple:
    ui = list(pd.unique(df.ui))
    samples = len(ui)
    if verbose:
        print(f"unit unique identifier (ui): {ui}")
        print(f"number of units: {samples}")

    train_cnt = int(samples * train_pct)
    val_cnt = int(samples * val_pct)
    test_cnt = int(samples * test_pct) + 1
    if verbose:
        print(f"train, val, test set counts: {train_cnt}, {val_cnt}, {test_cnt}")

    assert train_cnt + val_cnt + test_cnt == samples, "error"

    train_ui = random.sample(ui, train_cnt)
    ui = list(set(ui) - set(train_ui))
    val_ui = random.sample(ui, val_cnt)
    ui = list(set(ui) - set(val_ui))
    test_ui = random.sample(ui, test_cnt)
    ui = list(set(ui) - set(test_ui))

    assert len(ui) == 0, "error"

    if verbose:
        print(f"train ui: {train_ui}")
        print(f"val ui: {val_ui}")
        print(f"test ui: {test_ui}")

    train_df = df[df['ui'].isin(train_ui)]
    val_df = df[df['ui'].isin(val_ui)]
    test_df = df[df['ui'].isin(test_ui)]

    train_y = np.array(train_df.y, dtype=np.float32)
    val_y = np.array(val_df.y, dtype=np.float32)
    test_y = np.array(test_df.y, dtype=np.float32)

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
            @brief: retrieves a secret stored in AWS Secrets Manager. Requires AWS CLI and IAM user profile properly configured.

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





















