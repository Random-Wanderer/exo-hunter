import math
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer
from scipy.signal import butter, filtfilt

from ExoHunter.params import *

class Trainer():
    def __init__(self) -> None:
        pass

    def low_pass_filter(self, data, n_order=5, cutoff_frac=0.3, sample_freq=1/1800):
        nyquist = sample_freq / 2 # 0.5 times the sampling frequency
        # cutoff_hours = 1/cutoff_frac*nyquist*3600
        b, a = butter(n_order, cutoff_frac, btype='lowpass')
        data_filtered = filtfilt(b, a, data)
        return pd.DataFrame(data_filtered)

    def fourier_transform(self, data):
        '''
        Performs FFT on a dataframe and returns a dataframe
        '''
        data_fft = pd.DataFrame(np.abs(np.fft.fft(data, axis=1)))
        data_fft = data_fft.iloc[:, :(data_fft.shape[1] // 2)]
        return data_fft

    def pred_round(self, data, threshold=0.5):
        if data>threshold:
            return 1
        return 0

    def confusion_mat(self, y_true, y_pred, columns=CONFUSION_COLS, indices=CONFUSION_INDS):
        matrix = confusion_matrix(y_true, y_pred)
        return pd.DataFrame(matrix, columns=columns, index=indices)
