"""
Checkers Board Engine
=====================
Implements the full game state and rules for standard American Checkers (8x8).

Board representation:
    - 8x8 grid stored as list[list[int]]
    - Row 0 is the TOP of the board (Black's home), row 7 is the BOTTOM (White's home)
    - Only dark squares (where (row + col) is odd) are playable
    - Piece constants: EMPTY=0, BLACK=1, WHITE=2, BLACK_KING=3, WHITE_KING=4

Movement rules:
    - Black pieces move DOWN (increasing row index)
    - White pieces move UP (decreasing row index)
    - Kings move in all four diagonal directions
    - Jumps (captures) are mandatory — if a jump exists, you MUST take it
    - Multi-jumps are supported (chaining consecutive captures in one turn)
    - Landing on the back rank promotes a piece to King; promotion ends the turn
"""

from typing import List, Tuple, Optional
import copy

# ── Piece Constants ──────────────────────────────────────────────────
EMPTY = 0
BLACK = 1       # Regular black piece (moves down)
WHITE = 2       # Regular white piece (moves up)
BLACK_KING = 3  # Promoted black piece (moves in all diagonals)
WHITE_KING = 4  # Promoted white piece (moves in all diagonals)


class CheckersBoard:
    """
    Represents the complete Checkers game state: the 8x8 grid and whose turn it is.

    Coordinate system:
        (0,0) = top-left corner
        Black starts in rows 0–2, White starts in rows 5–7
    """

    def __init__(self):
        """Initialize a fresh board with pieces in starting positions."""
        self.grid: List[List[int]] = [[EMPTY for _ in range(8)] for _ in range(8)]
        self.current_turn: int = WHITE  # White moves first (standard American Checkers)
        self._setup_initial_board()

    def _setup_initial_board(self):
        """Place 12 black and 12 white pieces in their starting positions on dark squares."""
        for row in range(3):
            for col in range(8):
                if (row + col) % 2 != 0:  # Dark squares only
                    self.grid[row][col] = BLACK

        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 != 0:
                    self.grid[row][col] = WHITE

    # ── Display ──────────────────────────────────────────────────────

    def print_board(self):
        """Print the board to the console for debugging. b/w = pieces, B/W = kings."""
        symbols = {EMPTY: ".", BLACK: "b", WHITE: "w", BLACK_KING: "B", WHITE_KING: "W"}
        for row in range(8):
            row_str = " ".join([symbols[p] for p in self.grid[row]])
            print(f"{row} {row_str}")
        print("  0 1 2 3 4 5 6 7")

    # ── Query Helpers ────────────────────────────────────────────────

    def get_piece(self, row: int, col: int) -> int:
        """Return the piece at (row, col), or EMPTY if out of bounds."""
        if 0 <= row < 8 and 0 <= col < 8:
            return self.grid[row][col]
        return EMPTY

    def is_valid_square(self, row: int, col: int) -> bool:
        """Return True if (row, col) is within the 8x8 board."""
        return 0 <= row < 8 and 0 <= col < 8

    def is_opponent_piece(self, row: int, col: int, player: int) -> bool:
        """Return True if the piece at (row, col) belongs to the opponent of `player`."""
        piece = self.get_piece(row, col)
        if piece == EMPTY:
            return False
        if player == BLACK:
            return piece in (WHITE, WHITE_KING)
        if player == WHITE:
            return piece in (BLACK, BLACK_KING)
        return False

    # ── Move Generation ──────────────────────────────────────────────

    def get_valid_moves(self, player: int) -> List[Tuple[Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]]:
        """
        Generate all legal moves for `player`.

        Returns a list of move tuples: (start_pos, end_pos, list_of_jumped_pieces)
        
        Mandatory jump rule: if ANY jump exists for ANY piece, only jumps are returned.
        Normal (non-capturing) moves are only available when no jumps exist.
        """
        jumps: List = []
        normal_moves: List = []

        for r in range(8):
            for c in range(8):
                piece = self.grid[r][c]

                # Skip squares that don't belong to this player
                if not ((player == BLACK and piece in (BLACK, BLACK_KING)) or
                        (player == WHITE and piece in (WHITE, WHITE_KING))):
                    continue

                # Always collect jumps for every piece
                piece_jumps = self._get_jumps_for_piece(r, c)
                jumps.extend(piece_jumps)

                # Collect normal moves (only used if no jumps exist anywhere)
                piece_moves = self._get_normal_moves_for_piece(r, c)
                normal_moves.extend(piece_moves)

        # Mandatory jump rule: if jumps exist, they must be taken
        return jumps if jumps else normal_moves

    def _get_directions(self, piece: int) -> List[Tuple[int, int]]:
        """
        Return the diagonal directions a piece can move in.
        
        Regular pieces move forward only; kings move in all 4 diagonals.
        Each direction is a (delta_row, delta_col) tuple.
        """
        if piece == BLACK:
            return [(1, -1), (1, 1)]          # Down-left, Down-right
        elif piece == WHITE:
            return [(-1, -1), (-1, 1)]        # Up-left, Up-right
        elif piece in (BLACK_KING, WHITE_KING):
            return [(1, -1), (1, 1), (-1, -1), (-1, 1)]  # All four diagonals
        return []

    def _get_normal_moves_for_piece(self, row: int, col: int) -> List:
        """
        Generate non-capturing moves for the piece at (row, col).
        
        A normal move is one diagonal step onto an empty square.
        Returns list of (start, end, []) tuples (empty captured list).
        """
        moves = []
        piece = self.grid[row][col]
        for dr, dc in self._get_directions(piece):
            new_r, new_c = row + dr, col + dc
            if self.is_valid_square(new_r, new_c) and self.grid[new_r][new_c] == EMPTY:
                moves.append(((row, col), (new_r, new_c), []))
        return moves

    def _get_jumps_for_piece(
        self, row: int, col: int,
        captured_so_far: List[Tuple[int, int]] = None,
        current_piece: int = None
    ) -> List:
        """
        Recursively find all possible jump sequences starting from (row, col).

        Multi-jumps are supported: after each capture, the function recurses to find
        further captures from the landing square. The `captured_so_far` list prevents
        jumping the same piece twice in a multi-jump chain.

        Kinging rule: if a regular piece lands on the back rank, it is promoted to king
        and the turn ends immediately (no further jumps from that position).

        Returns list of (start, end, captured) tuples, where:
            - start  = the ORIGINAL starting position of the piece
            - end    = the final landing position after all jumps in the chain
            - captured = list of all (row, col) positions of captured pieces
        """
        if captured_so_far is None:
            captured_so_far = []

        jumps = []
        # Use the tracked piece type (matters during multi-jump before kinging)
        piece = current_piece if current_piece is not None else self.grid[row][col]
        player = BLACK if piece in (BLACK, BLACK_KING) else WHITE

        for dr, dc in self._get_directions(piece):
            # The jumped square is one step in the direction
            jumped_r, jumped_c = row + dr, col + dc
            # The landing square is two steps in the direction
            land_r, land_c = row + 2 * dr, col + 2 * dc

            if not self.is_valid_square(land_r, land_c):
                continue

            is_opponent = self.is_opponent_piece(jumped_r, jumped_c, player)
            is_empty_landing = self.grid[land_r][land_c] == EMPTY
            not_already_captured = (jumped_r, jumped_c) not in captured_so_far

            if not (is_opponent and is_empty_landing and not_already_captured):
                continue

            new_captured = captured_so_far + [(jumped_r, jumped_c)]

            # Check for promotion (kinging) on back rank
            is_kinged = False
            next_piece = piece
            if piece == BLACK and land_r == 7:
                is_kinged = True
                next_piece = BLACK_KING
            elif piece == WHITE and land_r == 0:
                is_kinged = True
                next_piece = WHITE_KING

            if is_kinged:
                # Turn ends immediately upon promotion — no further jumps
                jumps.append(((row, col), (land_r, land_c), new_captured))
            else:
                # Recurse to find multi-jump continuations
                sub_jumps = self._get_jumps_for_piece(land_r, land_c, new_captured, next_piece)
                if sub_jumps:
                    # Chain: keep original start, use sub-jump's end and captured list
                    for sub_jump in sub_jumps:
                        jumps.append(((row, col), sub_jump[1], sub_jump[2]))
                else:
                    # No further jumps — this single capture is the complete move
                    jumps.append(((row, col), (land_r, land_c), new_captured))

        return jumps

    # ── Move Execution ───────────────────────────────────────────────

    def make_move(self, move: Tuple[Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]):
        """
        Apply a validated move to the board, mutating the game state.

        Steps:
            1. Move the piece from start to end
            2. Remove all captured pieces
            3. Promote to king if the piece reaches the back rank
            4. Switch the active turn to the other player
        """
        start_pos, end_pos, captured = move
        piece = self.grid[start_pos[0]][start_pos[1]]

        # 1. Move the piece
        self.grid[start_pos[0]][start_pos[1]] = EMPTY
        self.grid[end_pos[0]][end_pos[1]] = piece

        # 2. Remove captured pieces from the board
        for r, c in captured:
            self.grid[r][c] = EMPTY

        # 3. Promote to king if reaching the back rank
        if piece == BLACK and end_pos[0] == 7:
            self.grid[end_pos[0]][end_pos[1]] = BLACK_KING
        elif piece == WHITE and end_pos[0] == 0:
            self.grid[end_pos[0]][end_pos[1]] = WHITE_KING

        # 4. Switch turn
        self.current_turn = WHITE if self.current_turn == BLACK else BLACK

    # ── Board Utilities ──────────────────────────────────────────────

    def clone(self) -> 'CheckersBoard':
        """
        Create a deep copy of this board state.
        
        Used by the minimax algorithm to explore hypothetical future states
        without mutating the actual game board.
        """
        new_board = CheckersBoard()
        new_board.grid = copy.deepcopy(self.grid)
        new_board.current_turn = self.current_turn
        return new_board

    def evaluate(self) -> float:
        """
        Compute a heuristic score for the current board state.

        Positive score favors Black, negative score favors White.

        Formula:
            V(s) = W_P × (blacks − whites) + W_K × (black_kings − white_kings) + W_C × (center_blacks − center_whites)

        Weights:
            W_P = 1.0  (material: regular pieces)
            W_K = 1.5  (material: kings are worth 1.5× regular pieces)
            W_C = 0.5  (positional: controlling the center 4×2 zone)

        This is the fallback heuristic used when no CNN model is available.
        """
        W_P = 1.0   # Regular piece weight
        W_K = 1.5   # King weight (more valuable due to mobility)
        W_C = 0.5   # Center control bonus

        p_black = p_white = k_black = k_white = c_black = c_white = 0

        # Center zone: rows 3–4, cols 2–5 (the most influential squares)
        center_rows = {3, 4}
        center_cols = {2, 3, 4, 5}

        for r in range(8):
            for c in range(8):
                piece = self.grid[r][c]
                is_center = r in center_rows and c in center_cols

                if piece == BLACK:
                    p_black += 1
                    if is_center:
                        c_black += 1
                elif piece == BLACK_KING:
                    k_black += 1
                    if is_center:
                        c_black += 1
                elif piece == WHITE:
                    p_white += 1
                    if is_center:
                        c_white += 1
                elif piece == WHITE_KING:
                    k_white += 1
                    if is_center:
                        c_white += 1

        return W_P * (p_black - p_white) + W_K * (k_black - k_white) + W_C * (c_black - c_white)

    # ── Game Over Detection ──────────────────────────────────────────

    def check_game_over(self) -> Optional[int]:
        """
        Check whether the game has ended.

        Returns:
            BLACK (1) — Black wins (White has no moves)
            WHITE (2) — White wins (Black has no moves)
            None      — Game is still in progress

        A player who cannot make any legal move on their turn loses.
        This covers both "no pieces remaining" and "all pieces blocked".
        """
        moves = self.get_valid_moves(self.current_turn)
        if not moves:
            # Current player has no moves → they lose, opponent wins
            return WHITE if self.current_turn == BLACK else BLACK
        return None
