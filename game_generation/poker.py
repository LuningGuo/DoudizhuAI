import pickle
import numpy as np
from random import shuffle
from game_generation.cards import Cards


class Poker:
    def __init__(self):
        self.poker_dict = {'3': 4, '4': 4, '5': 4, '6': 4, '7': 4,
                           '8': 4, '9': 4, 'T': 4, 'J': 4, 'Q': 4,
                           'K': 4, 'A': 4, '2': 4, 'S': 1, 'B': 1}
        self.cards = Cards(np.array(list(self.poker_dict.values())))
        self.poker_str = ''.join([k * v for k, v in self.poker_dict.items()])

    @staticmethod
    def str_to_array(cards_string):
        init_dict = {'3': 0, '4': 0, '5': 0, '6': 0, '7': 0,
                     '8': 0, '9': 0, 'T': 0, 'J': 0, 'Q': 0,
                     'K': 0, 'A': 0, '2': 0, 'S': 0, 'B': 0}
        for i in cards_string:
            init_dict[i] += 1
        return np.array(list(init_dict.values()))

    def deal(self, pred_model):
        poker_str_list = list(self.poker_str)
        shuffle(poker_str_list)
        player1 = self.str_to_array(''.join(poker_str_list[0:17]))
        player2 = self.str_to_array(''.join(poker_str_list[17:34]))
        player3 = self.str_to_array(''.join(poker_str_list[34:51]))
        public = self.str_to_array(''.join(poker_str_list[51:54]))
        player_list = [player1, player2, player3]
        prob_list = [pred_model.predict_proba(i.reshape((1, -1)))[0, 1] for i in player_list]
        landlord = player_list.pop(np.argmax(prob_list))
        landlord = landlord + public
        peasant1 = player_list[0]
        peasant2 = player_list[1]
        return Cards(landlord), Cards(peasant1), Cards(peasant2)


if __name__ == '__main__':
    with open("../call_landlord/call_landlord_model.pkl", 'rb') as file:
        model = pickle.load(file)
    poker = Poker()
    l, p1, p2 = poker.deal(model)
    print(l, p1, p2, sep='\n')
