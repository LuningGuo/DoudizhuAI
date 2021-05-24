import pickle

from numpy import inf

from game_generation.hand_card import HandCard
from game_generation.minimax import minimax
from game_generation.poker import Poker
from game_generation.tree import Node


if __name__ == '__main__':
    # deal cards
    with open("../call_landlord/call_landlord_model.pkl", 'rb') as file:
        call_landlord_model = pickle.load(file)
    poker = Poker()
    l, p1, p2 = poker.deal(call_landlord_model)
    landlord = HandCard(l, is_landlord=True)
    peasant1 = HandCard(p1)
    peasant2 = HandCard(p2)

    # start game
    with open("../call_landlord/score_model.pkl", 'rb') as file:
        score_model = pickle.load(file)
    node = Node(current=landlord, up=peasant1, down=peasant2)
    while node is not None and not node.is_over():
        opt_value, opt_child = minimax(node, 3, -inf, +inf, score_model)
        node = opt_child

