import numpy as np


class Cards:
    def __init__(self, card_array=None):
        self.card_array = np.zeros(15, dtype=int) \
            if card_array is None else card_array

    def __len__(self):
        return len(self.card_array)

    def __sub__(self, other):
        result = Cards(self.card_array)
        result.card_array -= other.card_array
        return result

    def __add__(self, other):
        result = Cards(self.card_array)
        result.card_array += other.card_array
        return result

    def __str__(self):
        card_str = ['3', '4', '5', '6', '7', '8', '9',
                    'T', 'J', 'Q', 'K', 'A', '2', 'S', 'B']
        return ''.join([i * j for i, j in zip(card_str, self.card_array)])

    def copy(self):
        result = Cards(self.card_array)
        return result
