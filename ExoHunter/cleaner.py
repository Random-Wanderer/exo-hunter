import math
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split

from ExoHunter.params import *

class Cleaner():
    def __init__(self) -> None:
        pass

    def min_window(self, data):
        temp = data.drop(columns='LABEL').T
        minim = temp[temp>-660].count().min()
        return minim

    def get_kaggle_data(self, data_name, test_size=0.2, drive=1):
        train_path = os.path.join('..', DRIVE[drive], data_name, FILEPATHS[data_name][0])
        test_path = os.path.join('..', DRIVE[drive], data_name, FILEPATHS[data_name][1])
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        return train_data, test_data

    def get_nasa_data(self, data_name='nasa', test_size=0.2, drive=0):
        exo_path = os.path.join('..', DRIVE[drive], data_name, FILEPATHS[data_name][0][0])
        non_exo_path = os.path.join('..', DRIVE[drive], data_name, FILEPATHS[data_name][0][1])
        exo_data = pd.read_csv(exo_path)
        non_exo_data = pd.read_csv(non_exo_path)
        exo_data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], inplace=True)
        non_exo_data.drop(columns=['Unnamed: 0'], inplace=True)
        exo_data = exo_data.T
        non_exo_data = non_exo_data.T
        exo_data[['LABEL']] = 2
        non_exo_data[['LABEL']] = 1
        all_data = pd.concat([exo_data, non_exo_data])
        minim = self.min_window(all_data)
        labels = all_data[['LABEL']]
        all_data = all_data.iloc[:, :minim]
        all_data[['LABEL']] = labels
        train_data, test_data = train_test_split(all_data, test_size=test_size)
        return train_data, test_data

    def get_raw_data(self, data_name='nasa', test_size=0.2, drive=1):
        if data_name == 'kaggle':
            return self.get_kaggle_data(test_size=test_size, drive=drive)
        if data_name == 'nasa':
            return self.get_nasa_data(test_size=test_size, drive=drive)
        return None

    def get_proc_data(self, data_name='nasa', test_size=0.2, drive=0):
        train_path = os.path.join('..', 'processed_data', data_name, FILEPATHS[data_name][1][0])
        test_path = os.path.join('..', 'processed_data', data_name, FILEPATHS[data_name][1][1])
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        return train_data, test_data

    def get_data(self, data_name='nasa', test_size=0.2, drive=0, raw=0):
        if raw:
            return self.get_raw_data(data_name, test_size, drive)
        return self.get_proc_data(data_name, test_size, drive)

    def get_Xy(self, data):
        X = data.drop(columns='LABEL')
        y = data['LABEL'].map({1:0, 2:1})
        return X, y
