# -*- coding: utf-8 -*-

from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping

class RNNModel():
    # Class contains two runnable functions. A scaler and the run_model function.
    # The rnn_model is created to be used inside of the run_model function.
    def __init__(self):
        pass


    def scaler(self, X_train_data, X_test_data):
        # Scale X features of the data
        robust_scaler = RobustScaler()

        train_scale = robust_scaler.fit_transform(X_train_data)
        test_scale = robust_scaler.transform(X_test_data)

        return train_scale, test_scale


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


    def run_model(self, X_data, y_data, X_validate, y_validate):
        # Runs model on data. Need training and validation data to run
        model = self.rnn_model()
        es = EarlyStopping(patience=2, restore_best_weights=True)
        model.fit(X_data, y_data, epochs=50, batch_size=64, callbacks=[es],
                  validation_data= (X_validate, y_validate), validation_split=0.2)

        return model
