# Alexis' Ensemble Pipeline
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

from sklearn.utils.estimator_checks import check_estimator


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.formatter = Formatter()
        self.X_cnn = FunctionTransformer(lambda data: self.formatter.prep_data(data, rnn_model=False))
        self.X_rnn = FunctionTransformer(lambda data: self.formatter.prep_data(data, rnn_model=True))
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        cnn_pipeline = Pipeline([
            ('cnn_preparation', self.X_cnn),
            ('cnn_model', CNNModel().classifier())
        ])

        rnn_pipeline = Pipeline([
            ('rnn_preparation', self.X_rnn),
            ('rnn_model', RNNModel().classifier())
        ])

        self.pipeline = StackingClassifier(
            estimators=[('cnn', cnn_pipeline),
                        ('rnn', rnn_pipeline)],
            final_estimator=SVC(),
            cv=5
        )

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on test data"""
        y_pred = self.pipeline.predict(X_test)
        #y_prob = self.pipeline.predict_proba(X_test)
        #return Recall().update_state(y_true=y_test,y_pred=y_pred), Precision().update_state(y_true=y_test,y_pred=y_pred),y_pred#,y_prob
        return recall_score(y_true=y_test,y_pred=y_pred), precision_score(y_true=y_test,y_pred=y_pred),y_pred#,y_prob



if __name__ == "__main__":
    # get data
    train_data = pd.read_csv('raw_data/processed_data/nasaTrain.csv',index_col='KepID')
    test_data = pd.read_csv('raw_data/processed_data/nasaTest.csv',index_col='KepID')
    # set instanciate cleaner and formatter
    cleaner = Cleaner()
    formatter = Formatter()
    #Get X and y
    X,y = cleaner.get_Xy(train_data)
    X_test,y_test = cleaner.get_Xy(test_data)
    # Check if the format is right --> if not reshape it
    X = formatter.length_check(X)
    X_test = formatter.length_check(X_test)
    # hold out
    #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    # train
    trainer = Trainer(X, y)
    trainer.run()
    # evaluate
    recall,precision,pred = trainer.evaluate(X_test=X_test,y_test=y_test)
    print(recall,precision,pred)
