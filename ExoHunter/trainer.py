# Alexis' Ensemble Pipeline
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
import numpy as np
import pandas as pd


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        cnn_pipeline = Pipeline([
            ('cnn_preparation', Formatter.prep_data()),
            ('cnn_model', CNNModel.build_model())
        ])

        rnn_pipeline = Pipeline([
            ('rnn_preparation', Formatter.prep_data()),
            ('rnn_model', RNNModel.build_model())
        ])

        self.pipeline = StackingClassifier(
            estimators=[('cnn', cnn_pipeline),
                        ('rnn', rnn_pipeline)],
            final_estimator=SVC()
        )

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on test data"""
        y_pred = self.pipeline.predict(X_test)
        y_prob = self.pipeline.predict_proba(X_test)
        return Recall().update_state(y_true=y_test,y_pred=y_pred), Precision().update_state(y_true=y_test,y_pred=y_pred),y_prob



if __name__ == "__main__":
    # get data
    train_data = pd.read_csv('processed_data/nasa/nasaTrain.csv',index_col='KepID')
    test_data = pd.read_csv('processed_data/nasa/nasaTest.csv',index_col='KepID')
    # set X and y
    cleaner = Cleaner()
    X,y = cleaner.get_Xy(train_data)
    X_test,y_test = cleaner.get_Xy(test_data)
    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    # train
    trainer = Trainer(X_train, y_train)
    trainer.run()
    # evaluate
    recall,precision,probabilities = trainer.evaluate(X_test=X_test)
    print(recall,precision,probabilities)
