import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer
from scipy.signal import butter, filtfilt, savgol_filter
from pyts.preprocessing import InterpolationImputer

from ExoHunter.params import *

class Formatter():
    def __init__(self) -> None:
        pass

    # Prepping Methods
    def impute(self, data, missing_value=np.nan, strategy='linear'):
        imputer = InterpolationImputer(missing_value, strategy)
        return imputer.transform(data-1)

    def butter_filt(self, data, n_order=5, cutoff_frac=0.3):
        b, a = butter(n_order, cutoff_frac, btype='lowpass')
        data_filtered = filtfilt(b, a, data)
        return pd.DataFrame(data_filtered)

    def savgol_filt(self, data, window=15, n_order=5):
        filtered_data = savgol_filter(data, window, n_order)
        return filtered_data

    def fourier_transform(self, data):
        '''
        Performs FFT on a dataframe and returns a dataframe
        '''
        data_fft = pd.DataFrame(np.abs(np.fft.fft(data, axis=1)))
        data_fft = data_fft.iloc[:, :(data_fft.shape[1] // 2)]
        return data_fft

    def prep_data(self, data):
        data_ = self.impute(data-1)
        data_ = self.savgol_filt(data_)
        data_ = self.fourier_transform(data_)
        normer=Normalizer()
        data_ = normer.fit_transform(data_)
        return data_

    # Post Training methods
    def pred_round(self, data, threshold=0.5):
        if data>threshold:
            return 1
        return 0

    def confusion_mat(self, y_true, y_pred, columns=CONFUSION_COLS, indices=CONFUSION_INDS):
        matrix = confusion_matrix(y_true, y_pred)
        return pd.DataFrame(matrix, columns=columns, index=indices)
