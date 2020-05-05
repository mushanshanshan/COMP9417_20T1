import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt


def normalize(data):
    return (data - data.min()) / (data.max() - data.min())


def AUC_score(X, Y, model):
    _prob = model.predict_proba(X)[:,1]
    return roc_auc_score(Y, _prob)


def main():
    # Read data and creating test and training sets
    data = pd.read_csv("titanic.csv")
    data_normalized = normalize(data.iloc[:, :])
    training_set = data_normalized.iloc[:620, :]
    test_set = data_normalized.iloc[620:, :]
    training_set_x = training_set.iloc[:, :-1].values
    training_set_y = training_set.iloc[:, -1:].values
    test_set_x = test_set.iloc[:, :-1].values
    test_set_y = test_set.iloc[:, -1:].values

    # Part A
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(training_set_x, training_set_y)
    print('Part A(accuracy score for training dataset):', clf.score(training_set_x, training_set_y))
    print('Part A(accuracy score for test dataset):', clf.score(test_set_x, test_set_y))

    # Part B
    training_set_auc_score = []
    test_set_auc_score = []
    for i in range(2, 21):
        clf = tree.DecisionTreeClassifier(min_samples_leaf=i, random_state=1)
        clf.fit(training_set_x, training_set_y)
        
        training_set_auc_score.append(AUC_score(training_set_x, training_set_y, clf))
        test_set_auc_score.append(AUC_score(test_set_x, test_set_y, clf))
    print("\nPart B,The optimal number of min_samples_leaf is: ", test_set_auc_score.index(max(test_set_auc_score)) + 2)
    print("Part B,The AUC score is: ", max(test_set_auc_score))

    # Part C
    plt.bar(range(2, 21), training_set_auc_score, 0.4, color="blue")
    plt.ylim(0.8, 0.95)
    plt.xticks(range(2, 21))
    plt.xlabel("number of min_samples_leaf")
    plt.ylabel("AUC score")
    plt.title("AUC score for number of min_samples_leaf in training sets")
    plt.show()
    plt.bar(range(2, 21), test_set_auc_score, 0.4, color="blue")
    plt.ylim(0.8, 0.95)
    plt.xticks(range(2, 21))
    plt.xlabel("number of min_samples_leaf")
    plt.ylabel("AUC score")
    plt.title("AUC score for number of min_samples_leaf in test sets")
    plt.show()

    # Part D
    survived, total = 0, 0
    for index, row in data.iterrows():
        if row['Pclass'] == 1 & row['Sex'] == 1:
            total += 1
            if row['Survived'] == 1:
                survived += 1
    print("\nPart D, P(S=true | G=female, C=1): ", survived / total)


if __name__ == '__main__':
    main()
