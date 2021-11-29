import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer, StandardScaler
from scipy.signal import butter, filtfilt, savgol_filter
from pyts.preprocessing import InterpolationImputer

from .params import *

class Formatter():
    def __init__(self) -> None:
        '''
        Initialises Formatter
        '''
        # self.length_check(X)
        # self.prep_data(X)
        pass


    # Prepping Methods
    def impute(self, data, missing_value=np.nan, strategy='linear'):
        '''
        Imputes using linear interpolation any missing NaN values that occur
        as a result of the stitching of the raw data
        '''
        imputer = InterpolationImputer(missing_value, strategy)
        return imputer.transform(data - data.mean().mean())

    def butter_filt(self, data, n_order=5, cutoff_frac=0.3):
        '''
        Performs a Butterworth Low Pass Filtering of the input data
        '''
        b, a = butter(n_order, cutoff_frac, btype='lowpass')
        data_filtered = filtfilt(b, a, data)
        return pd.DataFrame(data_filtered)

    def savgol_filt(self, data, window=15, n_order=5):
        '''
        Implements a Savitzky–Golay Low Pass Filtering of the input data
        '''
        filtered_data = savgol_filter(data, window, n_order)
        return filtered_data

    def fourier_transform(self, data):
        '''
        Performs FFT on a dataframe and returns a dataframe
        '''
        data_fft = pd.DataFrame(np.abs(np.fft.fft(data, axis=1)))
        data_fft = data_fft.iloc[:, :(data_fft.shape[1] // 2)]
        return data_fft

    def normalize(self, data):
        '''
        Normalises the incoming light curve along its time axis
        '''
        normer = Normalizer()
        normalized_data = normer.fit_transform(data)
        return normalized_data

    def scaler(self,train_data):
        '''
        Scales the input light curve
        '''
        std = StandardScaler()
        scaled_data = std.fit_transform(train_data)
        return scaled_data

    def expand_shape(self, data):
        expand = np.expand_dims(data, -1)
        return expand

    def prep_data(self, data, rnn_model=False):
        '''
        Imputes, Filters, FFTs, and Normalizes the input data in one step
        '''
        data_ = self.impute(data)
        data_ = self.savgol_filt(data_)
        data_ = self.fourier_transform(data_)
        data_ = self.normalize(data_)
        if rnn_model:
            data_ = self.expand_shape(data_)
        return data_


    # Data Augmentation methods
    def repeat(self, data, default_len=DEFAULT_LEN):
        '''
        Repeats an incoming light curve, without smoothing, to a desired length
        '''
        new_len = data.iloc[0].shape[0]
        repeats = (default_len//new_len)+1
        repeated_data = np.tile(data, repeats)[:, :default_len]
        return pd.DataFrame(repeated_data)

    def mirror(self, data, default_len=DEFAULT_LEN):
        '''
        Mirrors an input light curve in time and repeats this mirrored
        curve to a desired length
        '''
        mirrored_data = pd.concat([data, data.iloc[:,::-1]], axis=1)
        mirrored_data = self.repeat(mirrored_data, default_len)
        return pd.DataFrame(mirrored_data)

    def mean_pad(self, data, default_len=DEFAULT_LEN):
        '''
        Pads the tail end of a light curve with the mean value of that
        given light curve to a desired length
        '''
        means = data.mean(axis=1)
        columns = default_len - data.shape[-1]
        means_df = pd.concat([means]*columns, axis=1, ignore_index=True)
        padded_data = pd.concat([data, means_df], axis=1, ignore_index=True)
        return padded_data

    def df_splitter(self, data, n_parts=3):
        '''
        Splits a DataFrame into n equal parts by row and returns a list of
        length n of dataframes, suitable for passing
        into the augmentation functions
        '''
        rows = data.shape[0]
        sub_rows = rows//n_parts
        data.sample(frac=1)
        data_ = []
        [data_.append(data.iloc[i*sub_rows:(i+1)*sub_rows]) for i in range(n_parts-1)]
        data_.append(data.iloc[(n_parts-1)*sub_rows:])
        return data_

    def augment(self, data, n_parts=3, default_len=DEFAULT_LEN):
        '''
        Calls the splitter and augmentation methods to augment training data
        '''
        data1, data2, data3 = self.df_splitter(data, n_parts)
        data1 = self.repeat(data1, default_len)
        data2 = self.mirror(data2, default_len)
        data3 = self.mean_pad(data3, default_len)
        data_ = pd.concat([data1, data2, data3])
        return data_

    def length_check(self, data, strategy='mirror', default_len=DEFAULT_LEN, n_parts=3):
        '''
        Checks the length of input light curve and extends/trims the curve
        appropriately using the selected strategy
        '''
        assert(isinstance(data, pd.DataFrame))
        if data.shape[-1] >= default_len:
            return data.iloc[:,:default_len]
        if strategy == 'mirror':
            return self.mirror(data, default_len)
        elif strategy == 'mean_pad':
            return self.mean_pad(data, default_len)
        elif strategy == 'repeat':
            return self.repeat(data, default_len)
        elif strategy == 'augment':
            return self.augment(data, n_parts, default_len)


    # Post Training methods
    def pred_round(self, data, threshold=0.5):
        '''
        Turns the floating point probabilities returned by the model into a
        binary categorical label
        '''
        if data>threshold:
            return 1
        return 0

    def confusion_mat(self, y_true, y_pred, columns=CONFUSION_COLS, indices=CONFUSION_INDS):
        '''
        Outputs a confusion matrix of Actual and Predicted positives and negatives
        '''
        matrix = confusion_matrix(y_true, y_pred)
        return pd.DataFrame(matrix, columns=columns, index=indices)
