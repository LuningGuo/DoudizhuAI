from numpy import inf


def minimax(node, depth, alpha, beta, pred_model):
    if depth == 0 or node.is_over():
        return node.get_state_value(pred_model), node
    if not node.current.is_landlord:
        max_value = -inf
        max_child = None
        for child in node.get_childern():
            value, _ = minimax(child, depth - 1, alpha, beta, pred_model)
            if value > max_value:
                max_value = value
                max_child = child
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return max_value, max_child
    else:
        min_value = +inf
        min_child = None
        for child in node.get_childern():
            value, _ = minimax(child, depth - 1, alpha, beta, pred_model)
            if value < min_value:
                min_value = value
                min_child = child
            beta = min(beta, value)
            if beta <= alpha:
                break
        return min_value, min_child
