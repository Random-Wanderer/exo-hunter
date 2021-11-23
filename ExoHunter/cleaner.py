import math
import numpy as np
import pandas as pd

from ExoHunter.params import *

class Cleaner():
    def __init__(self) -> None:
        pass

    def fourier_transform(df):
        df_fft = np.abs(np.fft.fft(df, axis=1))
        return df_fft
