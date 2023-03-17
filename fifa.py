import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from explorer import Explorer
from dtregression import DTRegressor


def skDTR(trainX, testX, x, trainY, testY, y):
    dtr = DecisionTreeRegressor()
    dtr.fit(trainX, trainY)
    y_pred = dtr.predict(testX)
    mse = mean_squared_error(testY, y_pred)
    print(f"\tMSE: {mse}")


def numberFromMoney(n):
    f = 1
    if 'K' in n:
        f = 1e3
    elif 'M' in n:
        f = 1e6
    if (f == 1):
        return float(n[1:])
    else:
        return float(n[1:-1]) * f


def toNumber(x):
    if '+' in x:
        return sum([int(num) for num in x.split('+')])
    elif '-' in x:
        nums = [int(num) for num in x.split('-')]
        return nums[0] - sum(nums[1:])
    else:
        return pd.to_numeric(x)


def fifa():
    data = pd.read_csv('CompleteDataset.csv', low_memory=False)
    exp = Explorer(data, "Potential", keep_columns=['Age', 'Overall', 'Value', 'Wage', 'Special', 'Acceleration',
                   "Dribbling", "Finishing", 'Agility', 'Ball control', 'Composure', 'Potential'], value_split=True)
    data = exp.data
    # Convert to numerical values
    exp.apply_func_column(numberFromMoney, "Value", "float")
    exp.apply_func_column(numberFromMoney, "Wage", "float")
    # Acceleration -> Nummber or operation
    exp.apply_func_column(toNumber, "Acceleration", "float")
    # Agility -> Number or operation
    exp.apply_func_column(toNumber, "Agility", "float")
    # Ball control -> Number or operation
    exp.apply_func_column(toNumber, "Ball control", "float")
    exp.apply_func_column(toNumber, "Finishing", "float")  # Finishing
    exp.apply_func_column(toNumber, "Dribbling", "float")  # Dribbling
    exp.apply_func_column(toNumber, "Composure", "float")  # Compsure
    trainX, testX, x, trainY, testY, y = exp.reinstantiate_x_y()
    data = exp.data
    high_corr, correlations_count = exp.correlation_in_dataset()
    # print(exp.check_balance(exp.target))
    dtr = DTRegressor(exp)
    dtr.fit()
    y_real = list(testY[exp.target].values)
    y_pred = dtr.predict(testX)
    acc = mean_squared_error(y_real, y_pred)
    print("FIFA: ")
    print(
        f"\tPrecision de R2, de la implementacion del potential en scratch: {acc}")
    skDTR(trainX, testX, x, trainY, testY, y)


if __name__ == "__main__":
    fifa()
