import pandas as pd
from sklearn import tree
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt


def min_max(df):
    return (df - df.min()) / (df.max() - df.min())

def search_optimal(x, y):
    l = []
    for i in range(2, 21):
        clf = tree.DecisionTreeClassifier(min_samples_leaf=i)
        clf = clf.fit(x_training, y_training)
        proba = clf.predict_proba(x)
        print(proba[:,1])
        sco = roc_auc_score(y, proba[:, 1])
        l.append(sco)
    return l

file = pd.read_csv('titanic.csv')
df = pd.DataFrame(file)

df_preprocessed = min_max(df)

x = df_preprocessed[['Pclass', 'Sex', 'Age', 'Siblings_Spouses_Aboard', 'Parents_Children_Aboard']]
y = df_preprocessed[['Survived']]

x_training = x[0: 620]
x_test = x[620: 887]
y_training = y[0: 620]
y_test = y[620:887]

# Part A
clf_trn = tree.DecisionTreeClassifier()
clf_trn = clf_trn.fit(x_training, y_training)
print(f'Score for training dataset is {clf_trn.score(x_training, y_training)}')
print(f'Score for test dataset is {clf_trn.score(x_test, y_test)}')

# Part B
sco_l_tst = search_optimal(x_test, y_test)
print(sco_l_tst)
opt_num_msl = sco_l_tst.index(max(sco_l_tst)) + 2
#print(f'The optimal number of min_samples_leaf is {opt_num_msl}')

# Part C
#sco_l_trn = search_optimal(x_training, y_training)
#print(sco_l_trn)

plt.title('AUC score for all iterations in training sets')
plt.xlabel('iteration')
plt.ylabel('AUC score')
plt.bar(range(2, 21), sco_l_trn)
plt.xticks(range(2, 21), range(2, 21))
plt.show()

plt.title('AUC score for all iterations in test sets')
plt.xlabel('iteration')
plt.ylabel('AUC score')
plt.bar(range(2, 21), sco_l_tst)
plt.xticks(range(2, 21), range(2, 21))
plt.show()

# Part D
survived_f1 = 0
for index, row in df_preprocessed.iterrows():
    if row['Pclass'] == 0 and row['Sex'] == 1 and row['Survived'] == 1:
        survived_f1 += 1
f1 = 0
for index, row in df_preprocessed.iterrows():
    if row['Pclass'] == 0 and row['Sex'] == 1:
        f1 += 1

print(f'P(S=true | G=female, C=1)={survived_f1 / f1}')
