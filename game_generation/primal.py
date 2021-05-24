class Primal:
    def __init__(self, cards, p_num=0, c_len=0, k_num=0, k_len=0, bomb=False):
        self.cards = cards
        self.p_num = p_num
        self.c_len = c_len
        self.k_num = k_num
        self.k_len = k_len
        self.bomb = bomb

    def __str__(self):
        return self.cards.__str__()

    def __len__(self):
        return sum(self.cards.card_array)
