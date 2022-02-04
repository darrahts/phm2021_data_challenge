from tensorflow import keras
import numpy as np
import utils
import os

























### NOTE - CURRENTLY USES LOCAL FILES, NOT DB
### TODO - REFACTOR FOR DATABASE USAGE
#
#
# def train(X_train: np.array = None,
#           y_train: np.array = None,
#           X_val: np.array = None,
#           y_val: np.array = None,
#           X_test: np.array = None,
#           y_test: np.array = None,
#           lr: float = .001,
#           batch_size: int = 512,
#           epochs: int = 300,
#           early_stop: bool = False,
#           monitor: str = 'val_loss',
#           mode: str = 'min',
#           min_delta: float = 24.0,
#           patience: int = 32,
#           directory: str = '',
#           input_name: str = '',
#           model: keras.Model = None,
#           verbose: bool = True):
#
#     assert model is not None, "[ERROR] must supply a model."
#     save_dir = directory+input_name+'_model/'
#
#     early_stopping = keras.callbacks.EarlyStopping(monitor=monitor,
#                                                    mode=mode,
#                                                    min_delta=min_delta,
#                                                    patience=patience,
#                                                    verbose=1,
#                                                    restore_best_weights=True)
#
#     model.compile(loss='mse',
#                   optimizer=keras.optimizers.Adam(lr=lr),
#                   metrics=[keras.metrics.RootMeanSquaredError()])
#
#     if X_val is not None and y_val is not None:
#         if early_stop == False:
#             history = model.fit(X_train, y_train,
#                                 batch_size=batch_size,
#                                 epochs=epochs,
#                                 validation_data=(X_val, y_val))
#         else:
#             history = model.fit(X_train, y_train,
#                                 batch_size=batch_size,
#                                 epochs=epochs,
#                                 validation_data=(X_val, y_val),
#                                 callbacks=[early_stopping])
#
#     elif early_stop == False:
#         history = model.fit(X_train, y_train,
#                             batch_size=batch_size,
#                             epochs=epochs)
#     else:
#         history = model.fit(X_train, y_train,
#                             batch_size=batch_size,
#                             epochs=epochs,
#                             callbacks=[early_stopping])
#
#     with open(directory + 'model_summary.txt', 'w') as f:
#         model.summary(print_fn=lambda x: f.write(x + '\n'))
#
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     if verbose:
#         print(f"[INFO] saving <{input_name}> model to <{directory + input_name}>")
#     if 'lstm' in directory:
#         # with open(directory+input_name+'_model/'+input_name+'.pkl', 'wb') as f:
#         #     pickle.dump(model, f)
#         model.save(save_dir + input_name + '_.h5')
#     else:
#         model.save(save_dir)
#
#
#
#     utils.make_plot(type='loss_plot',
#                     save_fig=1,
#                     X_val=X_val,
#                     y_val=y_val,
#                     history=history.history,
#                     directory=directory,
#                     input_name=input_name)
#
#     utils.make_plot(type='error_plot',
#                     save_fig=1,
#                     X_val=X_val,
#                     y_val=y_val,
#                     history=history.history,
#                     directory=directory,
#                     input_name=input_name)
#
#     if X_test is not None and y_test is not None:
#         evaluation = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
#
#     else:
#         X_test = X_train
#         y_test = y_train
#         evaluation = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
#
#     y_pred = model.predict(X_test)
#
#     utils.make_plot(type='rul_plot',
#                     save_fig=1,
#                     y_test=y_test,
#                     y_pred=y_pred,
#                     directory=directory,
#                     input_name=input_name)
#
#     # save the variables as a json dictionary to a text file
#     varstr = repr(locals())
#     with open(directory + 'locals.txt', "w") as f:
#         f.write(varstr)