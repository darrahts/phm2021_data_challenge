
import utils
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle



### NOTE - CURRENTLY USES LOCAL FILES, NOT DB
### TODO - REFACTOR FOR DATABASE USAGE

def prepare_for_training(df: pd.DataFrame = None,
                         lookback: int = 10,
                         horizon: int = 1,
                         n_out: int = 1,
                         save_files: bool = True,
                         directory: str = '',
                         verbose: bool = True):
    train_df, train_y, val_df, val_y, test_df, test_y = utils.train_test_split(df, verbose=True)
    cols = ['unit', 'Fc', 'hs', 'ui', 'y']
    if 'dataset' in df.columns:
        cols.append('dataset')
    if verbose:
        print(f"[INFO] popping <{cols}>")
    train_pop = pd.concat([train_df.pop(x) for x in cols], axis=1)
    val_pop = pd.concat([val_df.pop(x) for x in cols], axis=1)
    test_pop = pd.concat([test_df.pop(x) for x in cols], axis=1)

    n_features = len(train_df.columns)

    train_np = train_df.to_numpy()
    val_np = val_df.to_numpy()
    test_np = test_df.to_numpy()

    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler.fit(train_np)
    train_np = scaler.transform(train_np)
    val_np = scaler.transform(val_np)
    test_np = scaler.transform(test_np)

    X_train, y_train = utils.temporalize_data(train_np, train_y, lookback, horizon, n_features, n_out)
    X_test, y_test = utils.temporalize_data(test_np, test_y, lookback, horizon, n_features, n_out)
    X_val, y_val = utils.temporalize_data(val_np, val_y, lookback, horizon, n_features, n_out)

    if save_files:
        if verbose:
            print(f"[INFO] saving files to <{directory}>")

        train_df.to_csv(directory+'train_df.csv')
        val_df.to_csv(directory+'val_df.csv')
        test_df.to_csv(directory+'val_df.csv')

        np.save(directory+'train_y.npy', train_y)
        np.save(directory+'val_y.npy', val_y)
        np.save(directory+'test_y.npy', test_y)

        train_pop.to_csv(directory+'train_pop.csv')
        val_pop.to_csv(directory+'val_pop.csv')
        test_pop.to_csv(directory+'test_pop.csv')

        np.save(directory+'X_train.npy', X_train)
        np.save(directory+'X_val.npy', X_val)
        np.save(directory + 'X_test.npy', X_test)

        np.save(directory+'y_train.npy', y_train)
        np.save(directory+'y_val.npy', y_val)
        np.save(directory+'y_test.npy', y_test)

        pickle.dump(scaler, open(directory+'scaler.pkl', 'wb'))

    return X_train, y_train, X_val, y_val, X_test, y_test


def make():
    y_labels = []
    with open("data_csv/y_labels.txt", "r") as f:
        for l in f:
            y_labels.append(l.strip())

    t_labels = []
    with open("data_csv/t_labels.txt", "r") as f:
        for l in f:
            t_labels.append(l.strip())

    v_labels = []
    with open("data_csv/v_labels.txt", "r") as f:
        for l in f:
            v_labels.append(l.strip())


    files = [
    'data_csv/df08_all_resampled_interp_h0_fc1.csv',
    'data_csv/df08_set2.csv',
    'data_csv/df08_set3.csv',
    'data_csv/df08_set4.csv'
    ]

    directories = [
    'experiments/best_input_set/set1/',
    'experiments/best_input_set/set2/',
    'experiments/best_input_set/set3/',
    'experiments/best_input_set/set4/'
    ]

    # 50 timesteps back (our dataset has been downsampled by a factor of 5, this might need changed)
    lookback = 10

    # 1 timestep forward
    horizon = 1

    # rul
    n_out = 1

    for i in range(0, len(files)):
        df = pd.read_csv(files[i])
        df.drop(columns=[df.columns[0]], inplace=True)
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_for_training(df=df,
                                                                                lookback=lookback,
                                                                                horizon=horizon,
                                                                                n_out=n_out,
                                                                                save_files=True,
                                                                                directory=directories[i])

        print(X_train.shape)
        print(X_val.shape)
        print(X_test.shape)


