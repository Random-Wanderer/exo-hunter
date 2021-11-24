import math
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split

from ExoHunter.params import *

class Cleaner():
    def __init__(self) -> None:
        pass

    def get_data(self, data_name, set_='train', test_size=0.2):
        '''Function that returns train , val, and test dataframes
        for a given data path'''
        path = os.path.join('raw_data', data_name, FILEPATHS[data_name][TRAIN_OR_TEST[set_]])
        data = pd.read_csv(path)
        if set_=='train':
            data_train, data_val = train_test_split(data, test_size=test_size)
            return data_train, data_val
        return data
