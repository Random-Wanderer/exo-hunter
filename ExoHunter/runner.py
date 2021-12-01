import formatter
from tensorflow.keras.metrics import Recall, Precision
from rnn_model import RNNModel
from cnn_model import CNNModel
from formatter import Formatter
from cleaner import Cleaner
from params import DEFAULT_LEN
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import recall_score, precision_score
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy import stats
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D, Flatten, Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from params import DEFAULT_LEN
from sklearn.metrics import recall_score, precision_score
import matplotlib.pyplot as plt
import joblib



def build_model(activation1='relu',activation2='relu',learning_rate=0.0001,dropout_rate=0.4):
    model = Sequential()

    model.add(Conv1D(40, 8, padding='same', activation=activation1, input_shape=(DEFAULT_LEN//2,1)))
    model.add(MaxPool1D(4, padding='same'))
    model.add(Conv1D(40, 4, padding='same', activation=activation1))
    model.add(MaxPool1D(2, padding='same'))
    # model.add(Conv1D(40, 2, padding='same', activation=activation1))
    # model.add(MaxPool1D(2, padding='same'))

    model.add(Flatten())

    model.add(Dense(40, activation=activation2))
    model.add(Dropout(dropout_rate))
    model.add(Dense(30, activation=activation2))
    model.add(Dropout(dropout_rate))
    model.add(Dense(20, activation=activation2))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation=activation2))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate,beta_1=0.9,beta_2=0.999),
        loss='binary_crossentropy',
        metrics=['accuracy', Recall(), Precision()]
            )

    return model



def plot_history(history):
    plt.plot(history.history['loss'])
    plt.title('Train loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()


def plot_loss_accuracy(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='best')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='best')
    plt.show()

    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.title('Model Recall')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='best')
    plt.show()

    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.title('Model Precision')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='best')
    plt.show()



if __name__ == "__main__":
    # get data
    train_data = pd.read_csv('raw_data/processed_data/nasaTrain.csv',index_col='KepID')
    test_data = pd.read_csv('raw_data/processed_data/nasaTest.csv',index_col='KepID')
    #data = pd.read_csv('raw_data/nasaSelect7.csv',index_col='KepID')
    # set instanciate cleaner and formatter
    cleaner = Cleaner()
    formatter = Formatter()
    #Get X and y
    X,y = cleaner.get_Xy(train_data)
    X_test,y_test = cleaner.get_Xy(test_data)
    # Check if the format is right --> if not reshape it
    X = formatter.length_check(X)
    X = formatter.prep_data(X, rnn_model=False)
    model = build_model()
    es = EarlyStopping(patience=20,restore_best_weights=True)
    history = model.fit(X, y,
                        validation_split=0.2,
                        epochs=300,
                        batch_size=32,
                        verbose=1,
                        callbacks=[es])

    plot_loss_accuracy(history)

    X_test = formatter.prep_data(X_test, rnn_model=False)
    model.evaluate(X_test,y_test)
    joblib.dump(model,'model3.pkl')

    # batch_size = [16, 32, 64]
    # epochs = [10,20,30,40,50]
    # activation1 = ['relu','tanh']
    # activation2 = ['relu','tanh']
    # KC_cnn = KerasClassifier(build_fn= lambda: build_model, epochs=20, batch_size=32,validation_split = 0.2, callbacks=[EarlyStopping(patience=5)])
    # KC_cnn._estimator_type = "classifier"
    # search_grid = {'epochs':epochs,'batch_size':batch_size,'activation1':activation1,'activation2':activation2}#'build_fn__optimizer__hyper__learning_rate':adam_learning_rate,'build_fn__optimizer__hyper__beta_1':adam_beta1}
    # result = GridSearchCV(estimator=KC_cnn, param_grid=search_grid, scoring=('recall'), n_jobs=-1, cv=3)
    # grid_result = result.fit(X, y)
    # print(grid_result.best_estimator_)



# # get data
# train_data = pd.read_csv('raw_data/processed_data/nasaTrain.csv',index_col='KepID')
# test_data = pd.read_csv('raw_data/processed_data/nasaTest.csv',index_col='KepID')
# #data = pd.read_csv('raw_data/nasaSelect7.csv',index_col='KepID')
# # set instanciate cleaner and formatter
# cleaner = Cleaner()
# formatter = Formatter()
# #Get X and y
# X,y = cleaner.get_Xy(train_data)
# X_test,y_test = cleaner.get_Xy(test_data)
# # Check if the format is right --> if not reshape it
# X = formatter.length_check(X)
# X = formatter.prep_data(X, rnn_model=False)
# #X_test = formatter.length_check(X_test)
# # hold out
# #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
# # train
# #runner = CNNModel().classifier()


# model = build_model()
# es = EarlyStopping(patience=5)
# history = model.fit(X, y,
#                     validation_split=0.2,
#                     epochs=5,
#                     batch_size=32,
#                     verbose=0,
#                     callbacks=[es])

#model.evaluate(X_test,y_test)



# batch_size = [16, 32, 64]
# epochs = [10,20,30,40,50]
# activation1 = ['relu','tanh']
# activation2 = ['relu','tanh']
# adam_param_path = 'build_fn.optimizer._hyper'
# learning_rate = [0.0001,0.001,0.01,0.1]
# beta1 = [0.1,0.3,0.5,0.7,0.9]


# KC_cnn = KerasClassifier(build_fn= lambda: build_model, epochs=40, batch_size=32,validation_split = 0.2, callbacks=[EarlyStopping(patience=5)])
# KC_cnn._estimator_type = "classifier"
# search_grid = {'epochs':epochs,'batch_size':batch_size,'activation1':activation1,'activation2':activation2}#'build_fn__optimizer__hyper__learning_rate':adam_learning_rate,'build_fn__optimizer__hyper__beta_1':adam_beta1}
# result = GridSearchCV(estimator=KC_cnn, param_grid=search_grid, scoring=('recall'), n_jobs=-1, cv=3)
# grid_result = result.fit(X, y)
# print(grid_result.best_estimator_)
# # evaluate
# #recall,precision,pred = trainer.evaluate(X_test=X_test,y_test=y_test)
# #print(recall,precision,pred)
