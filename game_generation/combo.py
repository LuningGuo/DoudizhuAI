import numpy as np
from game_generation.cards import Cards


class Combo:
    def __init__(self, p, k=None):
        self.p = p
        self.k = k
        if k is None:
            self.cards = Cards(p.cards.card_array)
        else:
            self.cards = Cards(p.cards.card_array + k.cards.card_array)

    def __str__(self):
        if sum(self.cards.card_array > 0) == 0:
            return 'PASS'
        return self.cards.__str__()

    def __gt__(self, other):
        if not self.comparable(other):
            return False
        self_pos = np.where(self.p.cards.card_array > 0)[0]
        other_pos = np.where(other.p.cards.card_array > 0)[0]
        return self_pos[0] > other_pos[0]

    def __len__(self):
        return sum(self.cards.card_array)

    def comparable(self, other):
        if self.k is None:
            return other.k is None \
                   and self.p.c_len == other.p.c_len \
                   and self.p.p_num == other.p.p_num
        else:
            return other.k is not None \
                   and self.p.p_num == other.p.p_num \
                   and self.p.c_len == other.p.c_len \
                   and self.k.k_num == other.k.k_num


