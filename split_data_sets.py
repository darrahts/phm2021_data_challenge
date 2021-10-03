import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
import pandas as pd

#
#       set1: all parameters (xs, xv, w, 0)
#       set2: reduced parameters (2, xv, w, 0)
#       set3: no virtual reduced (2, w, 0)
#       set4: no virtual (xs, w, 0)
#

# df_all = 'df08_all.csv'
#
# utils.check_gpu()

#
# df, df_aux, y_labels, t_labels = utils.load_h5(fnames=utils.fnames,
#                                                sets=utils.sets,
#                                                h5_dir=utils.h5_dir,
#                                                csv_dir=utils.csv_dir,
#                                                save=2,
#                                                save_as=df_all,
#                                                verbose=True)
#
# # always get a reindexing duplicate axis error UNLESS
# # the dataframe is reloaded.... ?
# del df
# df = pd.read_csv(utils.csv_dir + df_all)
#
# df = utils.resample_df(df=df,
#                        factor=5,
#                        interp_method='time',
#                        csv_dir=utils.csv_dir,
#                        save=2,
#                        save_as='df08_all_resampled.csv')
#
# print(df.head())

#
# df = pd.read_csv(utils.csv_dir + 'df08_all_resampled.csv')
# df.drop(columns=[df.columns[0]], inplace=True)
# utils.interp_y(df=df,
#                csv_dir=utils.csv_dir,
#                save=1,
#                save_as='df08_all_resampled_interp.csv')
#
# del df
# df = pd.read_csv(utils.csv_dir + 'df08_all_resampled_interp.csv')
# df.drop(columns=[df.columns[0]], inplace=True)
# df = df[df.hs == 0]
# df.to_csv(utils.csv_dir + 'df08_all_resampled_interp_h0.csv')
#
# df = df[df.Fc == 1]
# df.to_csv(utils.csv_dir + 'df08_all_resampled_interp_h0_fc1.csv')


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

df = pd.read_csv('data_csv/df08_all_resampled_interp_h0_fc1.csv')
df.drop(columns=[df.columns[0]], inplace=True)
df.drop(columns=['dataset'], inplace=True)


# save each dataframe first

#       set1: all parameters (xs, xv, w, 0)
# set 1 is 'data_csv/df08_all_resampled_interp_h0_fc1.csv'

# #       set2: reduced parameters (2, xv, w, 0)

# df = pd.read_csv('data_csv/df08_all_resampled_interp_h0_fc1.csv')
# df.drop(columns=[df.columns[0]], inplace=True)
# df.drop(columns=['dataset'], inplace=True)
# _t_labels = t_labels[:]
# _t_labels.remove('P2')
# _t_labels.remove('P50')
# df.drop(columns=_t_labels, inplace=True)
# df.to_csv('df08_set2.csv')
#
#
# #       set3: no virtual, reduced parameters (2, w, 0)
# df.drop(columns=v_labels, inplace=True)
# df.to_scv('df08_set3.csv')

# #       set4: no virtual reduced (xs, w, 0)
# df = pd.read_csv('data_csv/df08_all_resampled_interp_h0_fc1.csv')
# df.drop(columns=[df.columns[0]], inplace=True)
# df.drop(columns=['dataset'], inplace=True)
# df.drop(columns=v_labels, inplace=True)
# df.to_scv('df08_set4.csv')



