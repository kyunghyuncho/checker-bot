from typing import List, Tuple, Optional
import copy

# Board piece constants
EMPTY = 0
BLACK = 1
WHITE = 2
BLACK_KING = 3
WHITE_KING = 4

class CheckersBoard:
    """
    Represents the Checkers Game State.
    The board is an 8x8 grid where pieces are placed on the dark squares.
    0,0 is the top-left corner.
    Black pieces move "down" the board (increasing row index).
    White pieces move "up" the board (decreasing row index).
    """

    def __init__(self):
        # Initialize an 8x8 grid with 0s (empty squares)
        # Using a list of lists for simplicity and ease of indexing
        self.grid = [[EMPTY for _ in range(8)] for _ in range(8)]
        self.current_turn = WHITE # White moves first
        self._setup_initial_board()

    def _setup_initial_board(self):
        """Places pieces in their starting positions."""
        # Black pieces take the top 3 rows
        for row in range(3):
            for col in range(8):
                if (row + col) % 2 != 0: # Place only on dark squares
                    self.grid[row][col] = BLACK
        
        # White pieces take the bottom 3 rows
        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 != 0:
                    self.grid[row][col] = WHITE

    def print_board(self):
        """Utility function to print the board state to the console."""
        symbols = {
            EMPTY: ".",
            BLACK: "b",
            WHITE: "w",
            BLACK_KING: "B",
            WHITE_KING: "W"
        }
        for row in range(8):
            row_str = " ".join([symbols[p] for p in self.grid[row]])
            print(f"{row} {row_str}")
        print("  0 1 2 3 4 5 6 7")

    def get_piece(self, row: int, col: int) -> int:
        """Helper to get a piece, returning EMPTY if out of bounds."""
        if 0 <= row < 8 and 0 <= col < 8:
            return self.grid[row][col]
        return EMPTY

    def is_valid_square(self, row: int, col: int) -> bool:
        """Check if a coordinate is within the board boundaries."""
        return 0 <= row < 8 and 0 <= col < 8

    def is_opponent_piece(self, row: int, col: int, player: int) -> bool:
        """Check if the piece at (row, col) belongs to the opponent."""
        piece = self.get_piece(row, col)
        if piece == EMPTY:
            return False
        if player == BLACK and piece in (WHITE, WHITE_KING):
            return True
        if player == WHITE and piece in (BLACK, BLACK_KING):
            return True
        return False

    def get_valid_moves(self, player: int) -> List[Tuple[Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]]:
        """
        Generates all valid moves for a given player.
        Returns a list of moves, where each move is a tuple:
        (start_pos, end_pos, list_of_jumped_pieces)

        Importantly, in Checkers, jumps are MANDATORY. 
        If a jump is available, normal moves cannot be played.
        """
        jumps = []
        normal_moves = []

        for r in range(8):
            for c in range(8):
                piece = self.grid[r][c]
                # Only check moves for the correct player's pieces
                if (player == BLACK and piece in (BLACK, BLACK_KING)) or \
                   (player == WHITE and piece in (WHITE, WHITE_KING)):
                   
                   # Find potential jumps for this piece
                   piece_jumps = self._get_jumps_for_piece(r, c)
                   jumps.extend(piece_jumps)

                   # Only look for normal moves if we haven't found any jumps globally yet
                   # (Optimization: could delay this, but simpler to calculate all)
                   if not jumps: 
                       piece_moves = self._get_normal_moves_for_piece(r, c)
                       normal_moves.extend(piece_moves)

        # Checkers Rule: If you can jump, you MUST jump.
        if len(jumps) > 0:
            return jumps
        return normal_moves

    def _get_directions(self, piece: int) -> List[Tuple[int, int]]:
        """Returns the valid move directions based on the piece type."""
        # Black moves down (+row), White moves up (-row). Kings move both.
        if piece == BLACK:
            return [(1, -1), (1, 1)]
        elif piece == WHITE:
            return [(-1, -1), (-1, 1)]
        elif piece in (BLACK_KING, WHITE_KING):
            return [(1, -1), (1, 1), (-1, -1), (-1, 1)]
        return []

    def _get_normal_moves_for_piece(self, row: int, col: int) -> List:
        """Calculates normal, non-jumping moves for a specific piece."""
        moves = []
        piece = self.grid[row][col]
        directions = self._get_directions(piece)

        for dr, dc in directions:
            new_r, new_c = row + dr, col + dc
            if self.is_valid_square(new_r, new_c) and self.grid[new_r][new_c] == EMPTY:
                # Format: (start, end, jumped_pieces)
                moves.append( ((row, col), (new_r, new_c), []) )
        return moves

    def _get_jumps_for_piece(self, row: int, col: int, captured_so_far: List[Tuple[int, int]] = None, current_piece: int = None) -> List:
        """
        Recursively calculates all multi-jump sequences for a piece.
        This represents the core complexity of Checkers move generation.
        """
        if captured_so_far is None:
            captured_so_far = []
        
        jumps = []
        # Use provided piece type (important for kinging mid-jump, though traditionally kinging ends the turn)
        piece = current_piece if current_piece is not None else self.grid[row][col] 
        player = BLACK if piece in (BLACK, BLACK_KING) else WHITE
        directions = self._get_directions(piece)

        for dr, dc in directions:
            # Check the skipped square and the landing square
            jumped_r, jumped_c = row + dr, col + dc
            land_r, land_c = row + 2 * dr, col + 2 * dc

            if self.is_valid_square(land_r, land_c):
                is_opponent = self.is_opponent_piece(jumped_r, jumped_c, player)
                is_empty_landing = self.grid[land_r][land_c] == EMPTY
                not_already_captured = (jumped_r, jumped_c) not in captured_so_far

                if is_opponent and is_empty_landing and not_already_captured:
                    new_captured = captured_so_far + [(jumped_r, jumped_c)]
                    
                    # We found a valid jump. We need to check for multi-jumps.
                    # Temporarily apply the jump to check for follow-ups
                    # We don't want to actually mutate the main board state deeply yet.
                    
                    # Check if the piece would become a king (ends turn in many standard rules, 
                    # but we will support continuing the jump if standard variation allows.
                    # For simplicity, standard US rules: if you land on the last row and aren't a king, 
                    # you become a king and your turn ENDS.
                    is_kinged = False
                    next_piece_state = piece
                    if piece == BLACK and land_r == 7:
                        is_kinged = True
                        next_piece_state = BLACK_KING
                    elif piece == WHITE and land_r == 0:
                        is_kinged = True
                        next_piece_state = WHITE_KING

                    if is_kinged:
                        # Turn ends upon kinging
                        jumps.append( ((row, col), (land_r, land_c), new_captured) )
                    else:
                        # Recursively find next jumps from this new landing square
                        # We must prevent moving back over the same piece, handled by `not_already_captured`
                        sub_jumps = self._get_jumps_for_piece(land_r, land_c, new_captured, next_piece_state)
                        
                        if sub_jumps:
                            # A multi-jump exists, so append those sequences instead of stopping here.
                            # The start coordinate remains the ORIGINAL start coordinate.
                            for sub_jump in sub_jumps:
                                # sub_jump is ((sub_start), (sub_end), (sub_captured))
                                # We want to map it to ((orig_start), (sub_end), combined_captured)
                                jumps.append( ((row, col), sub_jump[1], sub_jump[2]) )
                        else:
                            # No further jumps available, so this single jump forms a complete move
                            jumps.append( ((row, col), (land_r, land_c), new_captured) )
        
        return jumps

    def make_move(self, move: Tuple[Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]):
        """
        Applies a provided move to the board state.
        Handles moving the piece, capturing jumped pieces, and kinging.
        """
        start_pos, end_pos, captured = move
        piece = self.grid[start_pos[0]][start_pos[1]]

        # Move piece
        self.grid[start_pos[0]][start_pos[1]] = EMPTY
        self.grid[end_pos[0]][end_pos[1]] = piece

        # Remove captured pieces
        for r, c in captured:
            self.grid[r][c] = EMPTY

        # Handle Kinging
        if piece == BLACK and end_pos[0] == 7:
            self.grid[end_pos[0]][end_pos[1]] = BLACK_KING
        elif piece == WHITE and end_pos[0] == 0:
            self.grid[end_pos[0]][end_pos[1]] = WHITE_KING

        # Swap turns
        self.current_turn = WHITE if self.current_turn == BLACK else BLACK

    def clone(self) -> 'CheckersBoard':
        """Deep copies the board state, useful for search algorithms."""
        new_board = CheckersBoard()
        new_board.grid = copy.deepcopy(self.grid)
        new_board.current_turn = self.current_turn
        return new_board

    def evaluate(self) -> float:
        """
        Heuristic evaluation function as defined in PLAN.md.
        V(s) = w1(P_black - P_white) + w2(K_black - K_white) + w3(C_black - C_white) + w4(T_white - T_black)
        """
        # Weights (can be tuned or made configurable later)
        W_P = 1.0   # Piece weight
        W_K = 1.5   # King weight
        W_C = 0.5   # Center control weight
        W_T = 0.0   # Threats (complex to calculate perfectly, omitting for baseline heuristic, focusing on material and center)

        p_black = p_white = k_black = k_white = c_black = c_white = 0

        # Center squares (rows 3,4, cols 2,3,4,5)
        center_rows = {3, 4}
        center_cols = {2, 3, 4, 5}

        for r in range(8):
            for c in range(8):
                piece = self.grid[r][c]
                is_center = r in center_rows and c in center_cols

                if piece == BLACK:
                    p_black += 1
                    if is_center: c_black += 1
                elif piece == BLACK_KING:
                    k_black += 1
                    if is_center: c_black += 1
                elif piece == WHITE:
                    p_white += 1
                    if is_center: c_white += 1
                elif piece == WHITE_KING:
                    k_white += 1
                    if is_center: c_white += 1

        score = W_P * (p_black - p_white) + W_K * (k_black - k_white) + W_C * (c_black - c_white)
        return score

    def check_game_over(self) -> Optional[int]:
        """
        Checks if the game is over.
        Returns: 
        BLACK (1) if Black wins
        WHITE (2) if White wins
        0 for Draw (simplified: no moves left for current player means loss)
        None if game is still active
        """
        # A player loses if they have no valid moves on their turn
        # This implicitly covers having no pieces left.
        moves = self.get_valid_moves(self.current_turn)
        if not moves:
            # Current player cannot move, so the OTHER player wins
            return WHITE if self.current_turn == BLACK else BLACK
            
        return None # Game active
