# -*- coding: utf-8 -*-

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

class RNNModel():
    def __init__(self):
        pass

    def rnn_model(self):
        # Model runs Long short term memory(LSTM) RNN nueral network
        model = Sequential()

        model.add(layers.LSTM(30, activation='tanh', return_sequences=True))
        model.add(layers.Dropout(0.3))
        model.add(layers.LSTM(20, activation='tanh', return_sequences=True))
        model.add(layers.Dropout(0.3))
        model.add(layers.LSTM(10, activation='tanh', return_sequences=False))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        return model
