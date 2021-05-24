import pickle
from copy import deepcopy

import numpy as np
from itertools import combinations
from functools import reduce
from game_generation.cards import Cards
from game_generation.combo import Combo
from game_generation.kicker import Kicker
from game_generation.primal import Primal
from game_generation.poker import Poker


class HandCard:
    def __init__(self, cards, is_landlord=False):
        self.cards = cards
        self.is_landlord = is_landlord

    def __sub__(self, other):
        result = deepcopy(self)
        result.cards.card_array -= other.cards.card_array
        return result

    def __str__(self):
        if sum(self.cards.card_array) == 0:
            return 'Winner!'
        return self.cards.__str__()

    def __len__(self):
        return sum(self.cards.card_array)

    def get_next_combo(self, pre_c=None):
        if pre_c is None:
            return self.get_all_combo()
        c_all = self.get_all_combo()
        c_next = list()
        for c in c_all:
            if c > pre_c:
                c_next.append(c)
        return c_next

    def get_all_combo(self):
        c_list = list()
        p_list = self.get_all_primal()
        for p in p_list:
            if p.k_num == 0:
                c_list.append(Combo(p))
            else:
                k_list = self.get_kicker(p)
                for k in k_list:
                    c_list.append(Combo(p, k))
        return c_list

    def get_kicker(self, p):
        k_list = list()
        card_left = self.cards.card_array - p.cards.card_array
        pos = np.where(card_left >= p.k_num)[0]
        if len(pos) < p.k_len:
            return k_list
        combination = combinations(pos, p.k_len)
        for c in combination:
            card_array = np.zeros(15, dtype=int)
            card_array[np.array(c)] = p.k_num
            k = Kicker(Cards(card_array), p.k_num, p.k_len)
            k_list.append(k)
        return k_list

    def get_all_primal(self):
        solo = self.get_simple(p_num=1)
        solo_chain = self.get_chain(p_num=1)
        pair = self.get_simple(p_num=2)
        pair_chain = self.get_chain(p_num=2)
        trio = self.get_simple(p_num=3)
        trio_chain = self.get_chain(p_num=3)
        bomb = self.get_bomb()
        trio_solo = self.get_simple(p_num=3, k_num=1)
        trio_pair = self.get_simple(p_num=3, k_num=2)
        four_solo = self.get_simple(p_num=4, k_num=1)
        four_pair = self.get_simple(p_num=4, k_num=2)
        plane_solo = self.get_chain(p_num=3, k_num=1)
        plane_pair = self.get_chain(p_num=3, k_num=1)
        primal_all = reduce(
            lambda x, y: x + y,
            [solo, solo_chain, pair, pair_chain, trio,
             trio_chain, bomb, trio_solo, trio_pair,
             four_solo, four_pair, plane_solo, plane_pair]
        )
        return primal_all

    def get_simple(self, p_num, k_num=0):
        pos = np.where(self.cards.card_array >= p_num)[0]
        p_list = list()
        for i in pos:
            card_array = np.zeros(15, dtype=int)
            card_array[i] = p_num
            if k_num == 0:
                k_len = 0
            elif p_num == 4:
                k_len = 2
            else:
                k_len = 1
            p = Primal(Cards(card_array), p_num, 1, k_num, k_len)
            p_list.append(p)
        return p_list

    def get_chain(self, p_num, k_num=0):
        pos = np.where(self.cards.card_array[0:12] >= p_num)[0]
        min_len = {1: 5, 2: 3, 3: 2}[p_num]
        p_list = list()
        for i in range(0, len(pos)):
            for j in range(i, len(pos)):
                if j - i == pos[j] - pos[i] and j - i >= min_len - 1:
                    card_array = np.zeros(15, dtype=int)
                    card_array[pos[i]:pos[j] + 1] = p_num
                    c_len = j + 1 - i
                    k_len = c_len if k_num > 0 else 0
                    p = Primal(Cards(card_array), p_num, c_len, k_num, k_len)
                    p_list.append(p)
        return p_list

    def get_bomb(self):
        p_list = list()
        pos = np.where(self.cards.card_array == 4)[0]
        for i in pos:
            card_array = np.zeros(15, dtype=int)
            card_array[i] = 4
            p = Primal(Cards(card_array), bomb=True)
            p_list.append(p)
        if self.cards.card_array[-1] == 1 \
                and self.cards.card_array[-2] == 1:
            card_array = np.zeros(15, dtype=int)
            card_array[-1] = 1
            card_array[-2] = 1
            p = Primal(Cards(card_array), bomb=True)
            p_list.append(p)
        return p_list

    @staticmethod
    def get_pass():
        card_array = np.zeros(15, dtype=int)
        return [Primal(Cards(card_array))]


if __name__ == '__main__':
    with open("../call_landlord/call_landlord_model.pkl", 'rb') as file:
        model = pickle.load(file)
    poker = Poker()
    l, p1, p2 = poker.deal(model)
    landlord = HandCard(l, is_landlord=True)
    peasant1 = HandCard(p1)

    c_list = landlord.get_all_combo()
    for c in c_list:
        print('地主手牌：{}'.format(landlord))
        print('地主出：{}'.format(c))
        print('农民手牌：{}'.format(peasant1))
        print('农民可以出：', end='')
        print(' '.join([c.__str__() for c in peasant1.get_next_combo(c)]))
        print('-' * 50)
