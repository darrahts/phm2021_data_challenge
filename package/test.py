import os
import sys
import h5py
import numpy as np
import pandas as pd
base_dir = os.path.dirname(os.getcwd())
print(base_dir)
sys.path.insert(1, base_dir)
from package.api import DB as api
import package.utils as utils


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

for filename in fnames:
    print(filename)
    for _set in sets:
        print(_set)
        print(os.path.join(base_dir, h5_dir, filename))

sys.exit(0)