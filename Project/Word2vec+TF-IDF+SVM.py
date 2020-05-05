import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn import svm
from gensim.models.keyedvectors import KeyedVectors
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle

'''
reference: https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
'''


class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def load_topic_list():
    return {
        'FOREX MARKETS': [],
        'ARTS CULTURE ENTERTAINMENT': [],
        'BIOGRAPHIES PERSONALITIES PEOPLE': [],
        'DEFENCE': [],
        'DOMESTIC MARKETS': [],
        'HEALTH': [],
        'MONEY MARKETS': [],
        'SCIENCE AND TECHNOLOGY': [],
        'SHARE LISTINGS': [],
        'SPORTS': [],
        'IRRELEVANT': []
    }


'''
Calculate doc vector from word vector
'''


def process(model, case, topic, weight, words):
    dicc = {
        'FOREX MARKETS': 0,
        'ARTS CULTURE ENTERTAINMENT': 1,
        'BIOGRAPHIES PERSONALITIES PEOPLE': 2,
        'DEFENCE': 3,
        'DOMESTIC MARKETS': 4,
        'HEALTH': 5,
        'MONEY MARKETS': 6,
        'SCIENCE AND TECHNOLOGY': 7,
        'SHARE LISTINGS': 8,
        'SPORTS': 9,
        'IRRELEVANT': 10
    }

    case_list = np.zeros(300)
    count = 0
    for word in case:
        if word in model and word != '_':
            temp = model[word] * (weight[dicc[topic], words.index(word)] * 100)
            case_list = np.sum([case_list, temp], axis=0)
            count += 1
    return (case_list / count).tolist()


'''
calculate TF-IDF(word weight) by train_data or test_data
'''


def TF_IDF():
    dicc = load_topics_dict()

    model = KeyedVectors.load_word2vec_format('en.vec')

    train_data = pd.read_csv('train_pre.csv')
    test_data = pd.read_csv('test_pre.csv')
    for index, row in train_data.iterrows():
        dicc[row['topic']] += row['article_words'].split(',')
        print(index)
    for index, row in test_data.iterrows():
        dicc[row['topic']] += row['article_words'].split(',')
        print(index)
    for key, value in dicc.items():
        print(len(value))
        value = ' '.join(value)
        print(value)

    corpus = []

    for key, value in dicc.items():
        print(key)
        print(type(value))
        print(len(value))
        corpus.append(' '.join(value))

    print(len(corpus))

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    words = vectorizer.get_feature_names()
    weight = tfidf.toarray()

    train_list = []
    for index, row in train_data.iterrows():
        train_list.append(process(model, row['article_words'].split(','), row['topic'], weight, words))
        print(index)

    js = json.dumps(train_list)

    file = open('train_svm.json', 'w')
    file.write(js)
    file.close()

    test_list = []
    for index, row in test_data.iterrows():
        test_list.append(process(model, row['article_words'].split(','), row['topic'], weight, words))
        print(index)

    js = json.dumps(test_list)

    file = open('test_svm.json', 'w')
    file.write(js)
    file.close()


'''
Load train_set/test_set label from file
'''


def load_label():
    train_data = pd.read_csv('train_pre.csv')
    train_label = train_data['topic'].tolist()

    test_data = pd.read_csv('test_pre.csv')
    test_label = test_data['topic'].tolist()

    return train_label, test_label


'''
Load topic as dict
'''


def load_topics_dict():
    return {
        'FOREX MARKETS': 0,
        'ARTS CULTURE ENTERTAINMENT': 0,
        'BIOGRAPHIES PERSONALITIES PEOPLE': 0,
        'DEFENCE': 0,
        'DOMESTIC MARKETS': 0,
        'HEALTH': 0,
        'MONEY MARKETS': 0,
        'SCIENCE AND TECHNOLOGY': 0,
        'SHARE LISTINGS': 0,
        'SPORTS': 0,
        'IRRELEVANT': 0
    }


'''
Use GridSearchCV search the best parameters and draw hot colormap
'''


def GridSearchCV_test():
    train_label, test_label = load_label()

    with open('train_svm_idt.json', 'r') as file:
        train_list = json.load(file)

    with open('test_svm.json', 'r') as file:
        test_list = json.load(file)

    clf = svm.SVC(decision_function_shape='ovo')
    clf.set_params(kernel='rbf', probability=True)

    param_grid = {'C': np.logspace(-2, 10, 13), 'gamma': np.logspace(-9, 3, 13)}

    grid_search = GridSearchCV(clf, param_grid, n_jobs=10, verbose=10)
    grid_search.fit(train_list, train_label)

    scores = grid_search.cv_results_['mean_test_score'].reshape(len(param_grid['C']), len(param_grid['gamma']))

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot, norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(param_grid['C'])), param_grid['C'], rotation=45)
    plt.yticks(np.arange(len(param_grid['gamma'])), param_grid['gamma'])
    plt.title('Validation accuracy')
    plt.show()

    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)

    clf = svm.SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True).fit(
        train_list, train_label)
    predict = clf.predict(test_list)
    for i in range(500):
        if predict[i] != test_label[i]:
            print(i, ':', test_label[i], '---', predict[i])
    print("test acc:" + str(np.mean(predict == test_label)))


'''
Train a model by some spcific parameters
'''


def svm_clf():
    train_label, test_label = load_label()

    with open('train_svm_idt.json', 'r') as file:
        train_list = json.load(file)
    with open('test_svm_idt.json', 'r') as file:
        test_list = json.load(file)

    print(len(train_list))

    clf = svm.SVC(kernel='rbf', C=1000, gamma=0.001, probability=True).fit(train_list, train_label)

    with open('clf.pickle', 'wb') as f:
        pickle.dump(clf, f)

    predict = clf.predict(train_list)
    print("train data precision:" + str(np.mean(predict == train_label)))

    predict = clf.predict(test_list)
    print(classification_report(predict, test_label))
    # for i in range(500):
    # if predict[i] != test_label[i]:
    # print(i, ':', test_label[i], '---', predict[i])
    print("test data precision:" + str(np.mean(predict == test_label)))


'''
Tool function to print a dict
'''


def print_dict(dictt):
    for key, value in dictt.items():
        print(key, '---', value)


'''
Print the ten most relevent case
'''


def ten_best():
    with open('clf.pickle', 'rb') as f:
        clf = pickle.load(f)
    train_label, test_label = load_label()

    with open('test_svm_idt.json', 'r') as file:
        test_list = json.load(file)

    prob_y = clf.predict_proba(test_list)
    predict = clf.predict(test_list)

    topic_prob = load_topic_list()
    topic_index = load_topic_list()
    topic_sorted = load_topic_list()

    for i in range(500):
        topic_prob[predict[i]].append(max(prob_y[i]))
        topic_index[predict[i]].append(i)

    print(topic_index)

    for key, value in topic_sorted.items():
        value = np.argsort(topic_prob[key])[::-1]
        # print(value)
        print('\n')
        print(key, ':')
        for i in range(min(10, len(value))):
            print(topic_index[key][value[i]] + 9501, end=' ')


if __name__ == '__main__':
    ten_best()

    pass
