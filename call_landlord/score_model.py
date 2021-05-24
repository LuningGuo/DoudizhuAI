import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from game_generation.cards import Cards
from game_generation.hand_card import HandCard
from game_generation.poker import Poker


def get_primal_str(hc):
    p_list = hc.get_all_primal()
    p_str_all = list()
    for p in p_list:
        if len(p) + p.k_len * p.k_num <= 20 \
                and p.__str__() not in p_str_all:
            p_str_all.append(p.__str__())
    return p_str_all


def transform(data):
    card_array_list = list()
    for row in range(data.shape[0]):
        card_dict = {i: 0 for i in range(1, 16)}
        for col in range(data.shape[1]):
            card_dict[data[row, col]] += 1
        card_array = np.array(list(card_dict.values())).reshape((1, 15))
        card_array_list.append(card_array)
    return np.concatenate(card_array_list, axis=0)


def get_feature(card_array):
    hc = HandCard(Cards(card_array))
    poker_hc = HandCard(Poker().cards)
    p_str_all = get_primal_str(poker_hc)
    p_str = get_primal_str(hc)
    p_str_dict = {k: 1 if k in p_str else 0 for k in p_str_all}
    return np.array(list(p_str_dict.values())).reshape((1, -1))


if __name__ == '__main__':
    # transform data
    data = pd.read_csv('call_landlord_sample.csv', header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X_transformed = transform(X)

    # get features
    features = get_feature(X_transformed[0])
    for i in range(1, len(X_transformed)):
        print('\r', i, sep='', end='')
        feature = get_feature(X_transformed[i])
        features = np.concatenate([features, feature], axis=0)
    print('\n', features.shape, sep='')

    # training and validation
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.25)
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'Accuracy score: {accuracy_score(y_test, y_pred)}')

    # re-train on the whole dataset and save the model
    model = LogisticRegression(max_iter=500)
    model.fit(features, y)
    with open('score_model.pkl', 'wb') as file:
        pickle.dump(model, file)
