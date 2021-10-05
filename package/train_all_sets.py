
import utils
import pandas as pd
import numpy as np
import models
import training


# enable gpu memory growth
utils.check_gpu()


#       set1: all parameters (xs, xv, w, 0)
#       set2: reduced parameters (2, xv, w, 0)
#       set3: no virtual reduced (2, w, 0)
#       set4: no virtual (xs, w, 0)


names = [
'set1_xs_xv_w_0',
'set2_2_xv_w_0',
'set3_2_w_0',
'set4_xs_w_0'
]

directories = [
'experiments/best_input_set/set1/',
'experiments/best_input_set/set2/',
'experiments/best_input_set/set3/',
'experiments/best_input_set/set4/'
]

model_types = ['dnn/', 'cnn/', 'lstm/', 'bilstm/', 'aelstm/']


# 50 timesteps back (our dataset has been downsampled by a factor of 5, the downsampling rate is also a hyper parameter to be explored)
lookback = 10

j = 2

##### for loop block here
for i in range(1, len(names)):
    print(f"[INFO] training for {directories[i]+names[i]}")
    X_train = np.load(directories[i] + 'X_train.npy')
    X_val = np.load(directories[i] + 'X_val.npy')
    X_test = np.load(directories[i] + 'X_test.npy')

    y_train = np.load(directories[i] + 'y_train.npy')
    y_val = np.load(directories[i] + 'y_val.npy')
    y_test = np.load(directories[i] + 'y_test.npy')

    X_train = np.concatenate([X_train, X_val, X_test], axis=0)
    y_train = np.concatenate([y_train, y_val, y_test], axis=0)

    if j == 0: # dnn
        print("[INFO] getting DNN model")
        model = models.dnn(lookback=lookback,
                           n_features=X_train.shape[2],
                           n_out=y_train.shape[2])
        print("[INFO] training DNN")
        training.train(X_train=X_train,
                       y_train=y_train,
                       directory=directories[i] + model_types[j],
                       input_name=names[i],
                       model=model,
                       early_stop=True,
                       min_delta=.25,
                       patience=20)

    elif j == 1: # cnn
        print("[INFO] getting CNN model")
        model = models.cnn(lookback=lookback,
                           n_features=X_train.shape[2],
                           n_out=y_train.shape[2])
        print("[INFO] training CNN")
        training.train(X_train=X_train,
                       y_train=y_train,
                       directory=directories[i] + model_types[j],
                       input_name=names[i],
                       model=model)

    elif j == 2: # lstm
        print("[INFO] getting LSTM")
        model = models.lstm(lookback=lookback,
                           n_features=X_train.shape[2],
                           n_out=y_train.shape[2])
        print("[INFO] training LSTM")
        training.train(X_train=X_train,
                       y_train=y_train,
                       directory=directories[i] + model_types[j],
                       input_name=names[i],
                       model=model,
                       early_stop=True,
                       min_delta=.1,
                       patience=50,
                       monitor='loss',
                       batch_size=64,
                       epochs=1000,
                       lr=.0001)

    elif j == 3:
        pass
    elif j == 4:
        pass
    else:
        print('[ERROR] a valid model was not selected.')
        break




#### train function parameters
# def train(X_train: np.array = None,
#           y_train: np.array = None,
#           X_val: np.array = None,
#           y_val: np.array = None,
#           X_test: np.array = None,
#           y_test: np.array = None,
#           min_delta: int = 24,
#           patience: int = 32,
#           lr: float = .001,
#           batch_size: int = 512,
#           epochs: int = 300,
#           early_stop: bool = False,
#           directory: str = '',
#           input_name: str = '',
#           model: keras.Model = None):


