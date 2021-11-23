import math
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split

from ExoHunter.params import *

class Cleaner():
    def __init__(self) -> None:
        pass

    def get_data(self, data='kaggle', train_val_split=0.2):
        '''Function that returns train , val, and test dataframes
        for a given data path'''
        df_train_val = os.path.join('raw_data', *DATA_PATHS[data][0])
        df_test = os.path.join('raw_data', *DATA_PATHS[data][1])

        df_train, df_val = train_test_split(df_train_val, test_size=train_val_split)

        return df_train, df_val, df_test


    def fourier_transform(df):
        '''Performs FFT on a dataframe and returns a dataframe'''
        df_fft = np.abs(np.fft.fft(df, axis=1))
        return pd.DataFrame(df_fft)
