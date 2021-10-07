import numpy as np
import models
import training


### NOTE - CURRENTLY USES LOCAL FILES, NOT DB
### TODO - REFACTOR FOR DATABASE USAGE


dataset_dirs = [
    'experiments/lstm_rul_ds08/',
    'experiments/lstm_rul_ds01-07/'
]
di = 0

health_states = [
    'h0/',
    'h1/'
]
hi = 0

flight_classes = [
    'fc1/',
    'fc2/',
    'fc3/'
]
fi = 0

lookback = 10

for fi in range(0, len(flight_classes)):
    current_dir = directory = dataset_dirs[di] + health_states[hi] + flight_classes[fi]
    print(f"[INFO] current directory: <{current_dir}>")

    X_train = np.load(current_dir + 'X_train.npy')
    X_val = np.load(current_dir + 'X_val.npy')
    X_test = np.load(current_dir + 'X_test.npy')

    y_train = np.load(current_dir + 'y_train.npy')
    y_val = np.load(current_dir + 'y_val.npy')
    y_test = np.load(current_dir + 'y_test.npy')

    X_train = np.concatenate([X_train, X_val, X_test], axis=0)
    y_train = np.concatenate([y_train, y_val, y_test], axis=0)

    print("[INFO] getting LSTM model")
    model = models.lstm(lookback=lookback,
                        n_features=X_train.shape[2],
                        n_out=y_train.shape[2])

    print(f"[INFO] training <{current_dir}>")
    training.train(X_train=X_train,
                   y_train=y_train,
                   directory=current_dir,
                   input_name='',
                   model=model,
                   early_stop=True,
                   min_delta=.1,
                   patience=25,
                   monitor='loss',
                   batch_size=64,
                   epochs=300,
                   lr=.0001)