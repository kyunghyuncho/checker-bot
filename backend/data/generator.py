import os
import json
from typing import List, Dict
from backend.engine.board import CheckersBoard
from backend.engine.minimax import get_best_move

def serialize_board_state(board: CheckersBoard) -> List[List[int]]:
    """Converts the 2D CheckersBoard.grid into a JSON-serializable list of lists."""
    return [row[:] for row in board.grid]

def generate_synthetic_game(depth: int = 4, epsilon: int = 0.1, max_moves: int = 200) -> Dict:
    """
    Plays a single game of Checkers against itself using Minimax.
    Records every board state, the player whose turn it was, and the final result.
    """
    board = CheckersBoard()
    states = []
    
    # 1. Play the game
    move_count = 0
    while True:
        # Check game over before acting
        winner = board.check_game_over()
        if winner is not None:
            break
            
        if move_count > max_moves: # Prevent infinite loops in weird heuristic states
            winner = 0 # Draw
            break
            
        # Record the state BEFORE making a move
        states.append({
            "board": serialize_board_state(board),
            "turn": board.current_turn
        })
        
        move = get_best_move(board, depth, epsilon)
        
        if move:
            board.make_move(move)
            move_count += 1
        else:
            # Should be caught by check_game_over but fallback
            winner = 2 if board.current_turn == 1 else 1
            break
            
    # 2. Label the data
    # We backpropagate the final 'winner' to all recorded states
    # Labels: 1.0 for Black Win, 0.0 for White Win, 0.5 for Draw
    label = 0.5
    if winner == 1:
        label = 1.0
    elif winner == 2:
        label = 0.0
        
    for state in states:
        state["label"] = label
        
    return {
        "result": winner,
        "total_moves": move_count,
        "history": states
    }

def generate_dataset(num_games: int, output_file: str, depth: int = 4, epsilon: float = 0.1, progress_callback=None):
    """Generates a dataset of games and saves it to a JSON file."""
    dataset = []
    
    print(f"Generating {num_games} synthetic games (Depth: {depth}, Epsilon: {epsilon})...")
    
    for i in range(num_games):
        game_data = generate_synthetic_game(depth, epsilon)
        dataset.append(game_data)
        
        if (i+1) % 10 == 0 or (i+1) == num_games:
            print(f"Completed {i+1}/{num_games} games...")
            if progress_callback:
                progress_callback(i + 1, num_games)
            
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
        
    print(f"Dataset successfully saved to {output_file}.")

if __name__ == "__main__":
    # Provides a default way to test generation, but typically called from /api/generate
    generate_dataset(num_games=5, output_file="backend/data/test_dataset.json")
