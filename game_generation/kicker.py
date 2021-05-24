class Kicker:
    def __init__(self, cards, k_num=0, k_len=0):
        self.cards = cards
        self.k_num = k_num
        self.k_len = k_len

    def __str__(self):
        return self.cards.__str__()
