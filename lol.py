import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from explorer import Explorer
from dtclassifier import DTClassifier

def skDTC(trainX, testX, x, trainY, testY, y):
    dtc = DecisionTreeClassifier(max_depth=5)
    dtc.fit(trainX, trainY)
    y_pred = dtc.predict(testX)
    acc = accuracy_score(testY, y_pred)
    print("\tAccuracy of BlueWins prediction from scikit DTC: ", acc)

def lol():
    data = pd.read_csv('high_diamond_ranked_10min.csv')
    exp = Explorer(data, 'blueWins', value_split=True, keep_columns=['blueWins', "blueKills", "blueDeaths", "blueTowersDestroyed", "blueTotalGold", "redTowersDestroyed", "redTotalGold"])
    trainX, testX, x, trainY, testY, y = exp.reinstantiate_x_y()
    dtc = DTClassifier(exp, max_depth=5)
    dtc.fit()
    y_pred = dtc.predict(testX)    
    accu = exp.accuracy(y_pred)
    print("LOL: ")
    print(f"\tAccuracy of BlueWins prediction from scratch DTC: {accu}")
    skDTC(trainX, testX, x, trainY, testY, y)
    
if __name__ == "__main__":
    lol()
