import formatter
from formatter import Formatter
from cleaner import Cleaner
from params import DEFAULT_LEN
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

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
    X_test = formatter.length_check(X_test)
    X_test = formatter.prep_data(X_test, rnn_model=False)
    model = GradientBoostingClassifier(learning_rate=0.1,verbose=1,random_state=1,validation_fraction=0.2)
    model.fit(X, y)
    y_pred = model.predict(X_test)
    recall,precision,accuracy = recall_score(y_test,y_pred), precision_score(y_test,y_pred), accuracy_score(y_test,y_pred)
    print(recall,precision,accuracy)
