import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from nltk.corpus import stopwords
from nltk import download
import numpy as np
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
import time
import json
import sys

'''
Dump a dict to file
'''


def dump_dict(dictt, file_name):
    js = json.dumps(dictt)

    file = open(file_name + '.json', 'w')
    file.write(js)
    file.close()


'''
Calculate the numbre of words
'''


def count_vocab_num(data):
    word_dict = {}
    for index, row in data.iterrows():
        print(index)
        print(row['article_words'])
        for word in row['article_words']:
            if word not in word_dict.keys():
                word_dict[word] = 1
            else:
                word_dict[word] += 1
    return word_dict


'''
Delete stop words
'''


def word_pre_process(word_list):
    processed_list = [word for word in word_list if word not in stopwords.words('english')]
    processed_list = ','.join(processed_list)
    return processed_list


'''
Split the raw data
'''


def data_pre_process(data):
    data = data.drop(['article_number'], axis=1)
    for index, row in data.iterrows():
        row['article_words'] = word_pre_process(row['article_words'].split(','))
    return data


'''
Read data from file
'''


def read_data_and_process():
    # Read data
    train_data = pd.read_csv('training.csv')
    test_data = pd.read_csv('test.csv')
    download('stopwords')
    data_pre_process(train_data).to_csv('train_pre.csv', index=False)
    data_pre_process(test_data).to_csv('test_pre.csv', index=False)


'''
Import pre_training wordvec
'''


def load_model():
    print('Start loading model!')
    gensim_model = KeyedVectors.load_word2vec_format('en.vec')
    # gensim_model.init_sims(replace=True)
    print('Model loaded!')
    return gensim_model


'''
Caldulate WMDistance
'''


def distance(sentence, df, model):
    distances = []
    topics = []
    for index, row in df.iterrows():
        distances.append(model.wmdistance(row['article_words'], sentence))
        topics.append(row['topic'])
        print(distances[-1], ' --- ', topics[-1])
    return distances, topics


def test():
    dicc = {
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

    dis = np.load('dis.npy')
    top = np.load('top.npy')

    for i in range(dis.shape[0]):
        dicc[top[i]].append(dis[i])

    for key, value in dicc.items():
        print(key, '---', np.mean(value))


'''
Read preprocessed data
'''


def read_pre_data():
    train_data = pd.read_csv('train_pre.csv')
    for index, row in train_data.iterrows():
        row['article_words'] = row['article_words'].split(',')
    test_data = pd.read_csv('test_pre.csv')
    for index, row in test_data.iterrows():
        row['article_words'] = row['article_words'].split(',')
    return train_data, test_data


'''
Read WMDistance from file
'''


def load_distances():
    dis_dict = {}
    dis_dict = {}
    with open('distance_test_data.json', 'r') as file:
        dis_dict.update(json.load(file))
    return dis_dict


'''
Read label from file
'''


def load_label():
    train_data = pd.read_csv('train_pre.csv')
    train_label = train_data['topic'].tolist()

    test_data = pd.read_csv('test_pre.csv')
    test_label = test_data['topic'].tolist()

    return train_label, test_label


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
KNN model with weights
'''


def weighted_KNN(distances, labels, K, b=2, c=3):
    topics = load_topics_dict()

    sorted_index = np.argsort(distances)

    for i in range(K):
        topics[labels[sorted_index[i]]] += label_weight[labels[sorted_index[i]]] * math.e ** (
                -(distances[sorted_index[i]] - b) ** 2 / (2 * c ** 2))

    return topics


'''
KNN model without weights
'''


def KNN(distances, labels, K):
    topics = load_topics_dict()

    sorted_index = np.argsort(distances)

    for i in range(K):
        topics[labels[sorted_index[i]]] += 1

    return topics


'''
Calculate accuracy with different K, B and C
'''


def test_K_B_C(K, B_list, C_list):
    distances = load_distances()
    train_label, test_label = load_label()

    result_B_list, result_C_list, result_error_list = [], [], []

    for b in B_list:
        for c in C_list:
            error = 0
            for key, value in distances.items():
                predict_dict = weighted_KNN(value, train_label, K, b, c)
                predict = max(predict_dict, key=predict_dict.get)
                if (predict != test_label[int(key)]):
                    error += 1
            result_B_list.append(b)
            result_C_list.append(c)
            result_error_list.append(error / 500)
            print('B = ', b, ', C = ', c, ' completed!')

    return result_B_list, result_C_list, result_error_list


'''
Calculate accuracy with different K
'''


def test_K(max_K):
    distances = load_distances()

    train_label, test_label = load_label()

    KNN_error = []
    wKNN_error = []

    for K in range(2, max_K):
        error = 0
        for key, value in distances.items():

            predict_dict = KNN(value, train_label, K)
            predict = max(predict_dict, key=predict_dict.get)
            if (predict != test_label[int(key)]):
                error += 1
        KNN_error.append(error / 500)

    for K in range(2, max_K):
        error = 0
        for key, value in distances.items():

            predict_dict = weighted_KNN(value, train_label, K)
            predict = max(predict_dict, key=predict_dict.get)
            if (predict != test_label[int(key)]):
                error += 1
        wKNN_error.append(error / 500)

    return KNN_error, wKNN_error


def test_error(K):
    distances = load_distances()
    train_label, test_label = load_label()

    for key, value in distances.items():
        predict_dict = KNN(value, train_label, K)
        predict = max(predict_dict, key=predict_dict.get)
        if (predict != test_label[int(key)]):
            print(key, test_label[int(key)])
            print_dict(predict_dict)
            print('\n--------------------\n')


def print_dict(dictt):
    for key, value in dictt.items():
        print(key, '---', value)


'''
Plot accuracy with different K
'''


def plot_test_K(max_K):
    KNN_error, wKNN_error = test_K(max_K)

    print(wKNN_error)

    l1 = plt.plot([i for i in range(len(KNN_error))], KNN_error, 'r--', label='KNN')
    l2 = plt.plot([i for i in range(len(wKNN_error))], wKNN_error, 'g--', label='weighted_KNN')

    plt.plot([i for i in range(len(KNN_error))], KNN_error, 'ro-', [i for i in range(len(wKNN_error))], wKNN_error,
             'g+-')
    plt.title('KNN and wKNN with different K')
    plt.xlabel('K')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


'''
Plot accuracy with different K, B and C
'''


def plot_test_K_B_C(K):
    B_list = np.arange(1, 5, 0.05).tolist()
    C_list = np.arange(0.1, 4, 0.05).tolist()

    result_B_list, result_C_list, result_error_list = test_K_B_C(K, B_list, C_list)

    fig = plt.figure()
    ax = Axes3D(fig)

    result_B_list = np.array(result_B_list)
    result_C_list = np.array(result_C_list)
    result_error_list = np.array(result_error_list)

    ax.set_title('Parameters in the Gaussian distribution - Loss with K = ' + str(K))
    ax.set_xlabel('The parameter b')
    ax.set_zlabel('Loss')
    ax.set_ylabel('The parameter c')

    ax.plot_trisurf(result_B_list, result_C_list, result_error_list, linewidth=0, antialiased=False, cmap='rainbow')

    plt.show()


'''
Make a predict with specific K, B and C
'''


def make_pridict(K, B, C):
    labels = list(load_topics_dict().keys())
    distances = load_distances()
    train_label, test_label = load_label()
    predict_label = []
    for key, value in distances.items():
        predict_dict = weighted_KNN(value, train_label, K, B, C)
        predict = max(predict_dict, key=predict_dict.get)
        predict_label.append(predict)

    f1_score_list = metrics.f1_score(test_label, predict_label, labels=labels, average=None)

    recall_score_list = metrics.recall_score(test_label, predict_label, labels=labels, average=None)

    accuracy_score_list = metrics.precision_score(test_label, predict_label, labels=labels, average=None)

    # for i in range(len(labels)):
    # print(labels[i])
    # print('acc ', accuracy_score_list[i], 'recall  ', recall_score_list[i], 'f1  ', f1_score_list[i])

    print(metrics.accuracy_score(test_label, predict_label))

    for i in range(500):
        if predict_label[i] != test_label[i]:
            print(i, ':', test_label[i], '---', predict_label[i])

    return 0


'''
Print the ten most relevent case
'''


def ten_top(K, B, C):
    distances = load_distances()
    train_label, test_label = load_label()
    topics = {label: [] for label in list(load_topics_dict().keys())}

    print(topics)

    for key, value in distances.items():
        predict_dict = weighted_KNN(value, train_label, K, B, C)
        predict_label = max(predict_dict, key=predict_dict.get)
        topics[predict_label][int(key)] = predict_dict[predict_label]
    print(topics)


if __name__ == '__main__':
    # read_data_and_process()

    start_time = time.time()

    argv = sys.argv
    start = int(argv[1])
    end = int(argv[2])

    print('This termianal calculating form ', start, ' to ', end)
    train_data, test_data = read_pre_data()

    model = load_model()

    print('Runtime: ', time.time() - start_time)

    distances = {}

    for test_index, test_row in test_data.iloc[start:end].iterrows():
        temp_distance_dict = {}
        start_time = time.time()
        temp_distance = []
        print('Start calculating test text ', test_index, '.....')
        for train_index, train_row in train_data.iterrows():
            if (train_index % 300 == 0):
                print('Distance ', train_index, ' completed!')
            temp_distance.append(model.wmdistance(test_row['article_words'], train_row['article_words']))
        distances[test_index] = temp_distance
        temp_distance_dict[test_index] = temp_distance
        dump_dict(temp_distance_dict, str(test_index))
        print('Calculating test text ', test_index, 'complete! Runtime: ', time.time() - start_time)

    dump_dict(distances, str(start) + '--' + str(end))
    print(distances)

    # word_dict = count_vocab_num(train_data)

    # dump_dict(word_dict)
    # word_dict = sorted(word_dict.items(), key=lambda item:item[1])

    # print('Loaded model, time = ', time.time() - start_time)
    # dis, top = distance(temp, train_data, model)
    # np.save('dis.npy', np.array(dis))
    # np.save('top.npy', np.array(top))

    pass
