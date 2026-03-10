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


def generate_synthetic_game(depth: int = 4, temperature: float = 1.0, max_moves: int = 200) -> Dict:
    """
    Play a complete game of checkers via AI self-play and record the result.

    The AI uses minimax with alpha-beta pruning. The temperature parameter
    controls softmax move sampling for diverse training data.

    Args:
        depth:       Minimax search depth (higher = stronger but slower)
        temperature: Softmax temperature (τ=0 greedy, higher = more random)
        max_moves:   Safety limit to prevent infinite games (declared a draw)

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

        # AI selects a move (softmax sampling for diversity)
        move = get_best_move(board, depth, temperature)

        if move:
            board.make_move(move)
            move_count += 1
        else:
            # Fallback: no move found (should be caught by check_game_over)
            winner = 2 if board.current_turn == 1 else 1
            break

    # ── Label all states with the game outcome ───────────────────────
    # Every board state in the game gets the same label based on who won.
    # Discounting (if desired) is applied at training time, not here.
    if winner == 1:
        label = 1.0    # Black won
    elif winner == 2:
        label = 0.0    # White won
    else:
        label = 0.5    # Draw

    for state in states:
        state["label"] = label

    return {
        "result": winner,
        "total_moves": move_count,
        "history": states
    }


def generate_dataset(num_games: int, output_file: str, depth: int = 4,
                     temperature: float = 1.0, progress_callback=None):
    """
    Generate a dataset of self-play games and save to a JSON file.

    Args:
        num_games:         Number of games to simulate
        output_file:       Path to write the resulting JSON dataset
        depth:             Minimax search depth per game
        temperature:       Softmax temperature for move sampling
        progress_callback: Optional function(games_done, total_games) for UI updates
    """
    dataset = []

    print(f"Generating {num_games} synthetic games (Depth: {depth}, Temperature: {temperature})...")

    for i in range(num_games):
        game_data = generate_synthetic_game(depth, temperature)
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
