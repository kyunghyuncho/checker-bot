import random
from typing import Tuple, Optional, List
from backend.engine.board import CheckersBoard, BLACK, WHITE

def minimax(position: CheckersBoard, depth: int, alpha: float, beta: float, maximizing_player: bool) -> Tuple[float, Optional[Tuple]]:
    """
    Minimax search with Alpha-Beta pruning for Checkers.
    Returns: (best_score, best_move)
    """
    # Base cases: depth limit reached or game over
    winner = position.check_game_over()
    if winner is not None:
        if winner == BLACK: return float('inf'), None
        if winner == WHITE: return float('-inf'), None
        return 0, None # Draw

    if depth == 0:
        return position.evaluate(), None

    if maximizing_player:
        max_eval = float('-inf')
        best_move = None
        # In our representation, Black is the maximizing player (+ heuristic score)
        valid_moves = position.get_valid_moves(BLACK)
        
        # If no moves available, it's a loss, but we let check_game_over handle that cleanly higher in the tree
        if not valid_moves:
             return position.evaluate(), None
             
        for move in valid_moves:
            # Simulate the move
            child_state = position.clone()
            child_state.make_move(move)

            # Recurse: next turn is minimizing player (White)
            eval_score, _ = minimax(child_state, depth - 1, alpha, beta, False)
            
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
                
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break # Alpha-Beta Pruning: Minimize branch
        return max_eval, best_move
    else:
        min_eval = float('inf')
        best_move = None
        # White is the minimizing player (- heuristic score)
        valid_moves = position.get_valid_moves(WHITE)

        if not valid_moves:
             return position.evaluate(), None

        for move in valid_moves:
            # Simulate
            child_state = position.clone()
            child_state.make_move(move)

            eval_score, _ = minimax(child_state, depth - 1, alpha, beta, True)
            
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
                
            beta = min(beta, eval_score)
            if beta <= alpha:
                break # Alpha-Beta Pruning: Maximize branch
        return min_eval, best_move

def get_best_move(position: CheckersBoard, depth: int, epsilon: float = 0.0) -> Optional[Tuple]:
    """
    Entry point for AI decision making.
    Uses epsilon-greedy strategy to sometimes pick sub-optimal moves for data diversity.
    """
    valid_moves = position.get_valid_moves(position.current_turn)
    if not valid_moves:
        return None
        
    # Epsilon-Greedy Strategy:
    # With probability epsilon, choose a random move from all valid legal moves.
    # This prevents deterministic games and provides wider state-space coverage for training.
    if random.random() < epsilon:
        return random.choice(valid_moves)
        
    # Otherwise, pick the optimal move using Minimax
    is_maximizing = position.current_turn == BLACK
    _, move = minimax(position, depth, float('-inf'), float('inf'), is_maximizing)
    return move
