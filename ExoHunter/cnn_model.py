from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D, Flatten, Dense, Dropout
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from params import DEFAULT_LEN


class CNNModel():
    def __init__(self) -> None:
        pass

    def build_model(self):
        model = Sequential()

        model.add(Conv1D(40, 8, padding='same', activation='relu', input_shape=(DEFAULT_LEN//2,1)))
        model.add(MaxPool1D(4, padding='same'))
        model.add(Conv1D(40, 4, padding='same', activation='relu'))
        model.add(MaxPool1D(2, padding='same'))
        #model.add(Conv1D(40, 2, padding='same', activation='relu'))
        #model.add(MaxPool1D(2, padding='same'))

        model.add(Flatten())

        #model.add(Dense(30, activation='relu'))
        #model.add(Dropout(0.1))
        #model.add(Dense(30, activation='relu'))
        #model.add(Dropout(0.1))
        model.add(Dense(20, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(10, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(
            optimizer=Adam(learning_rate=0.0001,beta_1=0.9,beta_2=0.999),
            loss='binary_crossentropy',
            metrics=['accuracy', Recall(), Precision()]
        )

        return model

    def classifier(self):
        model = self.build_model()
        KC_cnn = KerasClassifier(build_fn= lambda: model, epochs=200, batch_size=32,validation_split = 0.2, callbacks=[EarlyStopping(patience=20,restore_best_weights=True)])
        KC_cnn._estimator_type = "classifier"
        return KC_cnn
