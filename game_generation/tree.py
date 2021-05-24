import pickle
import random

import numpy as np
from numpy import inf

from call_landlord.score_model import get_feature
from game_generation.hand_card import HandCard
from game_generation.poker import Poker


class Node:
    def __init__(self, current, down, up, pre1=None, pre2=None):
        self.CARD_VALUE = np.array(list(range(1, 14)) + [20, 30])
        self.current = current
        self.down = down
        self.up = up
        self.pre1 = pre1
        self.pre2 = pre2

    def __str__(self) -> str:
        string = 'Up: \t{} \nCurr: \t{} \nDown: \t{} \nPre1: \t{} \nPre2: \t{}'.format(
            self.up, self.current, self.down, self.pre1, self.pre2)
        return string

    def get_childern(self):
        children = list()
        pre = self.pre2 if self.pre1 is None else self.pre1
        next_list = self.current.get_next_combo(pre)
        if len(next_list) == 0:
            child = Node(current=self.down,
                         down=self.up,
                         up=self.current,
                         pre1=None,
                         pre2=self.pre1)
            children.append(child)
            return children
        for c in next_list:
            child = Node(current=self.down,
                         down=self.up,
                         up=self.current - c,
                         pre1=c,
                         pre2=self.pre1)
            children.append(child)
        return children

    def get_state_value(self, pred_model):
        hc_list = [self.current, self.down, self.up]
        landlord_index = np.where([hc.is_landlord for hc in hc_list])[0][0]
        landlord = hc_list.pop(landlord_index)
        peasant1 = hc_list[0]
        peasant2 = hc_list[1]
        landlord_score = self.get_score(landlord, pred_model)
        peasant1_score = self.get_score(peasant1, pred_model)
        peasant2_score = self.get_score(peasant2, pred_model)
        return (peasant1_score + peasant2_score)/2 - landlord_score

    @staticmethod
    def get_score(hc, pred_model):
        if len(hc) == 0:
            return +inf
        feature = get_feature(hc.cards.card_array).reshape((1, -1))
        score = pred_model.predict_proba(feature)[0, 1]/len(hc)
        return score

    def is_over(self):
        return len(self.current) == 0 or len(self.down) == 0 or len(self.up) == 0


if __name__ == '__main__':
    with open("../call_landlord/call_landlord_model.pkl", 'rb') as file:
        model = pickle.load(file)
    poker = Poker()
    l, p1, p2 = poker.deal(model)
    landlord = HandCard(l, is_landlord=True)
    peasant1 = HandCard(p1)
    peasant2 = HandCard(p2)

    node = Node(landlord, peasant1, peasant2)
    print(node)
    print('-' * 50)
    while not node.is_over():
        children = node.get_childern()
        next_index = random.randint(0, len(children) - 1)
        node = children[next_index]
        print(node)
        print('-' * 50)
