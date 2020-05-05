import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


def normalize(data):
    return (data - data.min()) / (data.max() - data.min())


def costFunction(x, y, theta):
    return np.dot((np.dot(x, theta) - y).T, (np.dot(x, theta) - y))

def targetFunction(x, theta):
    return theta[0] + x * theta[1]


def SGDtraining(X, Y, theta = np.array([-1, -0.5]).T, alpha=0.01, iterations_max=50):
    iteration_times = 0
    m, n = X.shape
    loss_list = []
    while iteration_times < iterations_max:
        loss = 0
        j = random.randint(0, len(X) - 1)
        
        theta = theta+alpha*X[:,iteration_times].T.dot((Y-X.dot(theta)))

        for i in range(m):
            loss += (Y[i] - targetFunction(X[j][1], theta)) ** 2
        loss = loss / len(X)
        loss_list.append(loss)
        iteration_times += 1
        print(theta, '----', loss)
    plt.plot(range(iterations_max), loss_list)
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.show()
    return theta

def SGDplot(X, Y):
    plt.title("SGD")
    for i in range(300):
        plt.scatter(X[i][1], Y[i])
    plt.show()


data = pd.read_csv("house_prices.csv")

data = data.drop(['No'], axis=1)

data_Norm = normalize(data)
data_Norm.insert(0, 'One', 1)

training_Data = data_Norm[0:300]
test_Data = data_Norm[300:]

X = training_Data[['One', 'house age']].values
Y = training_Data[['house price of unit area']].values

theta = SGDtraining(X, Y)

SGDplot(X, Y)




