import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Recall, Precision

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix

class CNNModel():
    def __init__(self, input_shape=(3197, 1)) -> None:
        self.input_shape = input_shape

    def build_model(self):
        model = Sequential()

        model.add(Conv1D(20, 4, padding='same', activation='relu', input_shape=self.input_shape)) #ticket
        model.add(MaxPool1D(2, padding='same'))
        model.add(Conv1D(40, 8, padding='same', activation='relu'))
        model.add(MaxPool1D(2, padding='same'))
        model.add(Conv1D(40, 8, padding='same', activation='relu'))
        model.add(MaxPool1D(2, padding='same'))

        model.add(Flatten())

        model.add(Dense(30, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(30, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(20, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(10, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', Recall(), Precision()]
        )

        return model
