import math
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split

from ExoHunter.params import *

class Cleaner():
    def __init__(self) -> None:
        pass

    def get_data(self, data_name, test_size=0.2):
        train_path = os.path.join('raw_data', data_name, FILEPATHS[data_name][TRAIN_OR_TEST["train"]])
        test_path = os.path.join('raw_data', data_name, FILEPATHS[data_name][TRAIN_OR_TEST["test"]])
        train_data = pd.read_csv(train_path)
        data_test = pd.read_csv(test_path)
        data_train, data_val = train_test_split(train_data, test_size=test_size)
        return data_train, data_val, data_test
