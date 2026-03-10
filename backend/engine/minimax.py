"""
Minimax AI Engine with Alpha-Beta Pruning
==========================================
Implements the game tree search algorithm for Checkers.

The algorithm treats Black as the MAXIMIZING player (positive evaluation = good for Black)
and White as the MINIMIZING player (negative evaluation = good for White).

Alpha-Beta pruning dramatically reduces the search space by skipping branches
that cannot possibly affect the final decision. This is critical for playable
performance at search depths of 4+.

The epsilon-greedy wrapper adds controlled randomness for data generation and
AI vs AI variety.
"""

import random
from typing import Tuple, Optional
from backend.engine.board import CheckersBoard, BLACK, WHITE


def minimax(
    position: CheckersBoard,
    depth: int,
    alpha: float,
    beta: float,
    maximizing_player: bool,
    model: Optional['CheckersLightningModule'] = None
) -> Tuple[float, Optional[Tuple]]:
    """
    Minimax search with Alpha-Beta pruning.

    Args:
        position:           Current board state to evaluate
        depth:              Remaining search depth (0 = leaf node → evaluate)
        alpha:              Best score the maximizer can guarantee (starts at -∞)
        beta:               Best score the minimizer can guarantee (starts at +∞)
        maximizing_player:  True if it's Black's turn (maximizer)
        model:              Optional trained CNN model to use for heuristic evaluation

    Returns:
        (score, move) — the best evaluation score and the move that produces it.
        The move is None at leaf nodes or terminal states.
    """
    # ── Terminal state: someone won ──────────────────────────────────
    winner = position.check_game_over()
    if winner is not None:
        if winner == BLACK:
            return float('inf'), None   # Black wins → maximizer's best outcome
        if winner == WHITE:
            return float('-inf'), None  # White wins → minimizer's best outcome
        return 0, None                  # Draw (shouldn't happen in standard checkers)

    # ── Depth limit reached: use heuristic evaluation ────────────────
    if depth == 0:
        if model is not None:
            # Import here to avoid circular dependencies
            import torch
            from backend.model.cnn import board_to_tensor
            
            tensor = board_to_tensor(position.grid, position.current_turn)
            tensor = tensor.unsqueeze(0)  # Add batch dimension: (1, 5, 8, 8)
            with torch.no_grad():
                # The model outputs probabilities: (p_black_win, p_white_win)
                # We need a scalar evaluation where positive = good for Black
                p_black, p_white = model(tensor)
                # Scale up to roughly match heuristic magnitudes [-20, +20]
                eval_score = (p_black.item() - p_white.item()) * 20.0
            return eval_score, None
            
        # Fallback to hardcoded piece-counting heuristic
        return position.evaluate(), None

    # ── Maximizing player (Black) ────────────────────────────────────
    if maximizing_player:
        max_eval = float('-inf')
        best_move = None
        valid_moves = position.get_valid_moves(BLACK)

        if not valid_moves:
            # No moves available — evaluate the losing position
            return position.evaluate(), None

        for move in valid_moves:
            # Simulate the move on a cloned board
            child_state = position.clone()
            child_state.make_move(move)

            # Recurse: next turn belongs to the minimizer (White)
            eval_score, _ = minimax(child_state, depth - 1, alpha, beta, False, model)

            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move

            # Alpha-Beta pruning: if this branch can't improve on what the
            # minimizer already has, stop searching siblings
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break

        return max_eval, best_move

    # ── Minimizing player (White) ────────────────────────────────────
    else:
        min_eval = float('inf')
        best_move = None
        valid_moves = position.get_valid_moves(WHITE)

        if not valid_moves:
            return position.evaluate(), None

        for move in valid_moves:
            child_state = position.clone()
            child_state.make_move(move)

            # Recurse: next turn belongs to the maximizer (Black)
            eval_score, _ = minimax(child_state, depth - 1, alpha, beta, True, model)

            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move

            # Alpha-Beta pruning: if this branch can't improve on what the
            # maximizer already has, stop searching siblings
            beta = min(beta, eval_score)
            if beta <= alpha:
                break

        return min_eval, best_move


def get_best_move(
    position: CheckersBoard, 
    depth: int, 
    epsilon: float = 0.0,
    model: Optional['CheckersLightningModule'] = None
) -> Optional[Tuple]:
    """
    Entry point for AI decision-making. Selects the best move for the current player.

    Uses an epsilon-greedy strategy:
        - With probability ε, pick a uniformly random legal move
        - Otherwise, use minimax search to pick the optimal move

    This randomness serves two purposes:
        1. During data generation: produces diverse training games
        2. During AI vs AI play: prevents identical games

    Args:
        position:  Current board state
        depth:     Search depth for minimax (higher = stronger but slower)
        epsilon:   Probability of choosing a random move instead of optimal

    Returns:
        A move tuple (start, end, captured), or None if no legal moves exist.
    """
    valid_moves = position.get_valid_moves(position.current_turn)
    if not valid_moves:
        return None

    # Epsilon-greedy: random exploration
    if random.random() < epsilon:
        return random.choice(valid_moves)

    # Optimal play: minimax with alpha-beta pruning
    # Black is maximizing (positive heuristic), White is minimizing (negative)
    is_maximizing = position.current_turn == BLACK
    _, move = minimax(position, depth, float('-inf'), float('inf'), is_maximizing, model)
    return move
