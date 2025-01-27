# -*- coding: utf-8 -*-

from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from params import DEFAULT_LEN


class RNNModel():
    # Class contains two runnable functions. A scaler and the run_model function.
    # The rnn_model is created to be used inside of the run_model function.
    def __init__(self):
        pass

    def build_model(self):
        model = Sequential()

        model.add(layers.LSTM(64, activation='tanh', recurrent_activation= 'tanh',
                              dropout=0.2, return_sequences= True, input_shape=(DEFAULT_LEN//2, 1)))#(rows, column)
        model.add(layers.LSTM(32, activation='tanh', recurrent_activation= 'tanh', dropout= 0.2,
                              return_sequences= False))
        model.add(layers.Dense(10, activation='relu', use_bias= True))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1, activation='hard_sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer= 'rmsprop', metrics=['binary_accuracy'])

        return model

    def classifier(self):
        model = self.build_model()
        KC_rnn = KerasClassifier(build_fn= lambda: model, epochs=2, batch_size=256, validation_split=0.2, callbacks=[EarlyStopping(patience=3)])
        KC_rnn._estimator_type = "classifier"
        return KC_rnn



    # def scaler(self, X_train_data, X_test_data):
    #     # Scale X features of the data
    #     robust_scaler = RobustScaler()

    #     train_scale = robust_scaler.fit_transform(X_train_data)
    #     test_scale = robust_scaler.transform(X_test_data)

    #     return train_scale, test_scale


    # def run_model(self, X_data, y_data, X_validate, y_validate):
    #     # Runs model on data. Need training and validation data to run
    #     model = self.rnn_model()
    #     es = EarlyStopping(patience=2, restore_best_weights=True)
    #     model.fit(X_data, y_data, epochs=50, batch_size=64, callbacks=[es],
    #               validation_data= (X_validate, y_validate), validation_split=0.2)

    #     return model
