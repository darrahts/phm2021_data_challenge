import os
import sys
import h5py
import numpy as np
import pandas as pd
base_dir = os.getcwd()
print(base_dir)
sys.path.insert(1, base_dir)

from api import DB as api
import utils as utils



"""
    params is a dictionary of the below format, that MUST be implemented
    according to your own credentials. If you use AWS and secrets manager, 
    create a secret with the name as below. Alternatively, there are plenty
    of other ways to do this. 
    
    params = {'datasource.username': 'user1', 
              'datasource.password': 'mypassword', 
              'datasource.database': 'ncmapss_db', # <- NO CHANGE 
              'datasource.url': '192.168.1.123', # <- or a domain name
              'datasource.port': '5432'} # <- most likely keep the same
"""

params = {'datasource.username': 'darrahts', 
            'datasource.password': 'Darrah16', 
            'datasource.database': 'ncmapss_db', # <- NO CHANGE 
            'datasource.url': 'localhost', # <- or a domain name
            'datasource.port': '5432'} # <- most likely keep the same

#params = utils.get_aws_secret("/secret/ncmapssdb")
db, cur =  api.connect(params)
db.set_session(autocommit=True)
del params


asset_type = api._create_asset_type(asset_type='engine', subtype='ncmapss',
                                    description='turbine engine from N-CMAPSS dataset unit', db=db, cur=cur)

h5_dir = 'data_h5'
fnames = [
    'N-CMAPSS_DS01-005.h5',
    'N-CMAPSS_DS03-012.h5',
    'N-CMAPSS_DS04.h5',
    'N-CMAPSS_DS05.h5',
    'N-CMAPSS_DS06.h5',
    'N-CMAPSS_DS07.h5',
    'N-CMAPSS_DS08a-009.h5',
    'N-CMAPSS_DS08c-008.h5'
]

sets = ['dev', 'test']

df = pd.DataFrame()
asset_id = 1

assets = []
components = []

for filename in fnames:
    print(filename)
    for _set in sets:
        print(_set)
        with h5py.File(os.path.join(base_dir, h5_dir, filename), 'r') as hdf:
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

        df_a = pd.DataFrame(data=a_data, columns=a_labels)
        df_a['asset_id'] = -1
        df_a['dataset'] = filename.split('_')[1].split('.')[0]
        df_w = pd.DataFrame(data=w_data, columns=w_labels)
        df_x = pd.DataFrame(data=x_data, columns=x_labels)
        df_v = pd.DataFrame(data=v_data, columns=v_labels)
        df_t = pd.DataFrame(data=t_data, columns=t_labels)
        df_y = pd.DataFrame(data=y_data, columns=['y'])
        print(f"<{filename}> : {pd.unique(df_a.unit)}")
        for n in list(pd.unique(df_a.unit)):
            df_a.loc[df_a['unit'] == n, 'asset_id'] = asset_id
            asset_id = asset_id + 1

        df_temp = pd.concat([df_a, df_y, df_w, df_x, df_v, df_t], axis=1)
        # print(df_temp.head())
        if (len(df)) == 0:
            df = df_temp
        else:
            df = pd.concat([df, df_temp], axis=0)

        del df_a, df_w, df_x, df_v, df_t, df_y, a_data, w_data, t_data, x_data, y_data, df_temp

    df = df.round(5)
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

    serial_numbers = [utils.generate_serial_number(length=8) for _ in range(len(df_aux))]

    for i in range(0, len(df_aux)):
        asset = api._create_asset(type_id=int(asset_type.id.values[0]),
                                  common_name='ncmapss unit',
                                  age=float(df_aux.iloc[i].age),
                                  eol=float(df_aux.iloc[i].eol),
                                  rul=float(df_aux.iloc[i].eol - df_aux.iloc[i].age),
                                  units='cycles',
                                  serial_number=serial_numbers[i],
                                  db=db,
                                  cur=cur)
        print(asset)
        assets.append(asset)

        component = api._create_component(asset=asset,
                                          group_id=df_aux.iloc[i].group_id,
                                          unit=df_aux.iloc[i].unit,
                                          dataset=df_aux.iloc[i].dataset,
                                          db=db,
                                          cur=cur)
        print(component)
        components.append(component)

    start_id = api.execute("select max(id) from summary_tb;", db).values[0][0]
    if type(start_id) == type(None):
        start_id = 0
    df.index = pd.to_datetime(df.index, unit='s', origin='unix')
    df.index.names = ['dt']
    df.reset_index(inplace=True)
    df.index += start_id + 1
    df.index.names = ['id']
    df.reset_index(inplace=True)
    df.loc[:, 'dt'] = df.loc[:, 'dt'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    df.head()

    summary_cols = api.get_fields('summary_tb', as_list=True, db=db)
    print(f"summary_cols: {summary_cols}")

    api.batch_insert(df=df[summary_cols],
                     tb='summary_tb',
                     db=db,
                     cur=cur)

    telemetry_cols = api.get_fields('telemetry_tb', as_list=True, db=db)
    print(f"telemetry_cols: {telemetry_cols}")

    api.batch_insert(df=df[telemetry_cols],
                     tb='telemetry_tb',  # num_batches is optional with default value = 10
                     db=db,
                     cur=cur)  # verbose is optional with default value = False

    degradation_cols = api.get_fields('degradation_tb', as_list=True, db=db)
    print(f"degradation_cols: {degradation_cols}")

    api.batch_insert(df=df[degradation_cols],
                     tb='degradation_tb',
                     num_batches=10,
                     db=db,
                     cur=cur,
                     verbose=True)

    del df, df_aux, serial_numbers, start_id
    df = pd.DataFrame()

cur.close()
db.close()
