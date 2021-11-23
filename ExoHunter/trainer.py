import math
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.signal import butter, filtfilt

from ExoHunter.params import *

class Trainer():
    def __init__(self) -> None:
        pass

    def low_pass_filter(self, df, n_order=5, cutoff_frac=0.3, sample_freq=1/1800):
        '''
        Low pass filters the input lux curves to remove noise
        '''
        nyquist = sample_freq / 2
        cutoff_hours = 1/cutoff_frac*nyquist*3600
        b, a = butter(n_order, cutoff_frac, btype='lowpass')
        df_filtered = filtfilt(b, a, df)
        return pd.DataFrame(df_filtered)

    def fourier_transform(df):
        '''
        Performs FFT on a dataframe and returns a dataframe
        '''
        df_fft = np.abs(np.fft.fft(df, axis=1))
        return pd.DataFrame(df_fft)

    def pred_round(df, threshold=0.5):
        '''

        '''
        if df>threshold:
            return 1
        return 0

    def confusion_mat(y_true, y_pred, columns=CONFUSION_COLS, indices=CONFUSION_INDS):
        matrix = confusion_matrix(y_true, y_pred)
        return pd.DataFrame(matrix, columns=columns, index=indices)
