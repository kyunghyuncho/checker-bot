"""
Self-Play Data Generator for Checkers
======================================
Generates training data by having the minimax AI play games against itself.

Each generated game produces a sequence of board states, all labeled with the
game's final outcome. This "outcome supervision" approach means that every
position in a winning game is marked as favorable for the winner.

Output format (JSON):
    [
        {
            "result": 1,          # 1 = Black wins, 2 = White wins, 0 = Draw
            "total_moves": 47,
            "history": [
                {
                    "board": [[0, 1, 0, ...], ...],  # 8×8 grid
                    "turn": 2,                        # Whose turn it was
                    "label": 1.0                      # Outcome label
                },
                ...
            ]
        },
        ...
    ]

Labels:
    1.0 = Black won this game
    0.0 = White won this game
    0.5 = Draw (max_moves exceeded without a winner)
"""

import json
from typing import List, Dict

from backend.engine.board import CheckersBoard
from backend.engine.minimax import get_best_move


def serialize_board_state(board: CheckersBoard) -> List[List[int]]:
    """Convert the board's 8×8 grid to a JSON-serializable list of lists."""
    return [row[:] for row in board.grid]


def generate_synthetic_game(depth: int = 4, epsilon: float = 0.1, max_moves: int = 200, discount_factor: float = 0.0) -> Dict:
    """
    Play a complete game of checkers via AI self-play and record the result.

    The AI uses minimax with alpha-beta pruning. The epsilon parameter controls
    the probability of choosing a random move instead of the optimal one,
    which produces more diverse training data.

    Args:
        depth:           Minimax search depth (higher = stronger but slower)
        epsilon:         Probability of a random move (0 = fully deterministic)
        max_moves:       Safety limit to prevent infinite games (declared a draw)
        discount_factor: Exponential discounting factor γ ∈ [0, 1].
                         For position at distance d from game end:
                           label = 0.5 + (outcome − 0.5) × (1 − γ)^d
                         γ=0: no discounting (all positions get full outcome label)
                         γ=1: only the final position gets the true label

    Returns:
        Dict containing the game result, total moves, and full state history.
    """
    board = CheckersBoard()
    states = []
    move_count = 0

    # ── Self-play loop ───────────────────────────────────────────────
    while True:
        # Check if the game has ended (no legal moves for current player)
        winner = board.check_game_over()
        if winner is not None:
            break

        # Safety valve: declare a draw after too many moves
        if move_count > max_moves:
            winner = 0  # Draw
            break

        # Record the board state BEFORE making the move
        states.append({
            "board": serialize_board_state(board),
            "turn": board.current_turn
        })

        # AI selects a move (epsilon-greedy: sometimes random for diversity)
        move = get_best_move(board, depth, epsilon)

        if move:
            board.make_move(move)
            move_count += 1
        else:
            # Fallback: no move found (should be caught by check_game_over)
            winner = 2 if board.current_turn == 1 else 1
            break

    # ── Label all states with discounted game outcome ─────────────────
    # Outcome: 1.0 = Black won, 0.0 = White won, 0.5 = Draw
    if winner == 1:
        outcome = 1.0
    elif winner == 2:
        outcome = 0.0
    else:
        outcome = 0.5

    # Apply exponential discounting toward 0.5 (draw)
    # label_t = 0.5 + (outcome - 0.5) * (1 - γ)^d
    # where d = distance from end = (T - 1 - t)
    T = len(states)
    for t, state in enumerate(states):
        d = T - 1 - t  # distance from the final position
        discount = (1.0 - discount_factor) ** d
        state["label"] = 0.5 + (outcome - 0.5) * discount

    return {
        "result": winner,
        "total_moves": move_count,
        "history": states
    }


def generate_dataset(num_games: int, output_file: str, depth: int = 4,
                     epsilon: float = 0.1, progress_callback=None,
                     discount_factor: float = 0.0):
    """
    Generate a dataset of self-play games and save to a JSON file.

    Args:
        num_games:         Number of games to simulate
        output_file:       Path to write the resulting JSON dataset
        depth:             Minimax search depth per game
        epsilon:           Randomness level for move selection
        progress_callback: Optional function(games_done, total_games) for UI updates
        discount_factor:   Exponential discounting factor for labels (0 = no discounting)
    """
    dataset = []

    print(f"Generating {num_games} synthetic games (Depth: {depth}, Epsilon: {epsilon}, Discount: {discount_factor})...")

    for i in range(num_games):
        game_data = generate_synthetic_game(depth, epsilon, discount_factor=discount_factor)
        dataset.append(game_data)

        # Progress logging every 10 games or at completion
        if (i + 1) % 10 == 0 or (i + 1) == num_games:
            print(f"Completed {i + 1}/{num_games} games...")
            if progress_callback:
                progress_callback(i + 1, num_games)

    # Write dataset to disk
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"Dataset successfully saved to {output_file}.")


if __name__ == "__main__":
    # Standalone test: generate 5 games for quick verification
    generate_dataset(num_games=5, output_file="backend/data/test_dataset.json")
