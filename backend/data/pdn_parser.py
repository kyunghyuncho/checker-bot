import re
from typing import List, Dict, Tuple
from backend.engine.board import CheckersBoard, BLACK, WHITE

def parse_pdn(pdn_text: str) -> List[Dict]:
    """
    Parses a Portable Draughts Notation (PDN) file.
    Returns a list of parsed games.
    
    A PDN file typically contains headers like [Event "..."] 
    and then the move list like:
    1. 11-15 23-18 2. 8-11 ...
    
    For educational purposes and simplicity in this platform, 
    we extract the Result and the raw move sequence.
    """
    games = []
    
    # Split the file by empty lines or [Event] tags, roughly isolating games
    raw_games = re.split(r'(?=\[Event)', pdn_text)
    
    for raw_game in raw_games:
        if not raw_game.strip():
            continue
            
        game_data = {
            "headers": {},
            "moves": [],
            "result": None # 1.0 (Black Win), 0.0 (White Win), 0.5 (Draw)
        }
        
        # Parse Headers
        headers = re.findall(r'\[(.*?) "(.*?)"\]', raw_game)
        for key, val in headers:
            game_data["headers"][key] = val
            if key == "Result":
                if val == "1-0": game_data["result"] = 1.0
                elif val == "0-1": game_data["result"] = 0.0
                elif val == "1/2-1/2": game_data["result"] = 0.5
        
        # Parse Moves
        # Remove headers
        move_text = re.sub(r'\[.*?\]', '', raw_game)
        # Remove comments enclosed in {}
        move_text = re.sub(r'\{.*?\}', '', move_text)
        # Remove move numbers (e.g., "1.", "2.")
        move_text = re.sub(r'\d+\.', '', move_text)
        # Remove the result indicator from the end of move text if present
        move_text = move_text.replace('1-0', '').replace('0-1', '').replace('1/2-1/2', '').replace('*', '')

        # Tokenize by whitespace to get individual moves
        tokens = move_text.split()
        
        # PDN uses 1-32 tile notation. 
        # For our 8x8 grid representation, we map these to row, col.
        # Checkers playable squares (dark) are numbered 1-32 starting from top-left.
        for token in tokens:
            if '-' in token or 'x' in token:
                # Normal move (11-15) or Jump (11x18)
                game_data["moves"].append(token)

        games.append(game_data)
        
    return games

def pdn_square_to_rc(square_num: int) -> Tuple[int, int]:
    """
    Converts a PDN square number (1-32) to an 8x8 (row, col) coordinate.
    Note: PDN numbering logic can vary slightly in open-source checkers engines,
    but standard English Draughts starts 1 at top-left.
    """
    # 0-indexed internally
    idx = square_num - 1
    
    # Each row has 4 playable squares
    row = idx // 4
    
    # Calculate column based on whether the row is offset.
    # In our board representation, row 0 playable cols are 1, 3, 5, 7.
    # row 1 playable cols are 0, 2, 4, 6
    if row % 2 == 0:
        col = (idx % 4) * 2 + 1
    else:
        col = (idx % 4) * 2
        
    return row, col

def apply_pdn_moves_to_board(moves: List[str]) -> List[Tuple[CheckersBoard, str]]:
    """
    Replays a PDN game, generating board states at every step.
    Returns: List of tuples (BoardState, MoveNotation)
    """
    board = CheckersBoard()
    states = []
    
    for notation in moves:
        states.append((board.clone(), notation))
        
        # Parse the raw notation (e.g. 11-15)
        # This requires matching the start/end square against our move generator's outputs.
        # Since PDN doesn't explicitly list captured pieces in standard leaps (11x18),
        # we have to use our engine's get_valid_moves to find the *full* move tuple 
        # corresponding to these start/end positions.
        is_jump = 'x' in notation
        parts = notation.split('x') if is_jump else notation.split('-')
        
        try:
            start_sq = int(parts[0])
            end_sq = int(parts[-1]) # In multi-jumps (11x18x25), we just need start/end
            
            start_rc = pdn_square_to_rc(start_sq)
            end_rc = pdn_square_to_rc(end_sq)
            
            valid_moves = board.get_valid_moves(board.current_turn)
            
            # Find the move that matches these coordinates
            applied = False
            for v_move in valid_moves:
                vm_start, vm_end, _ = v_move
                if vm_start == start_rc and vm_end == end_rc:
                    board.make_move(v_move)
                    applied = True
                    break
                    
            if not applied:
                print(f"Warning: Move {notation} ({start_rc} to {end_rc}) not found in valid moves for turn {board.current_turn}")
                # Sometimes PDN numbering differs or a file has corrupted rules. 
                # We skip broken games in data processing.
                break
                
        except ValueError:
            print(f"Error parsing move token: {notation}")
            break
            
    return states
