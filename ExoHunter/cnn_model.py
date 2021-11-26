from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D, Flatten, Dense, Dropout
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping

class CNNModel():
    def __init__(self, input=(3197,1)) -> None:
        self.input_shape = input

    def build_model(self):
        model = Sequential()

        model.add(Conv1D(20, 4, padding='same', activation='relu', input_shape=self.input_shape))
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

        return KerasClassifier(build_fn=self.build_model, epochs=20, batch_size=32, callbacks=[EarlyStopping(patience=5)])
