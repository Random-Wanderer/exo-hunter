import math
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from exo-hunter.params import *

class Trainer():
    def __init__(self) -> None:
        pass

    def pred_round(df, threshold=0.5):
        if df>threshold:
            return 1
        return 0

    def confusion_mat(y_true, y_pred, columns=CONFUSION_COLS, indices=CONFUSION_INDS):
        matrix = confusion_matrix(y_true, y_pred)
        return pd.DataFrame(matrix, columns=columns, index=indices)
