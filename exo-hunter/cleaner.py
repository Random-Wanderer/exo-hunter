import math
import numpy as np
import pandas as pd

class Cleaner():
    def __init__(self) -> None:
        pass

    def pred_round(df, threshold=0.5):
        if df>threshold:
            return 1
        return 0
