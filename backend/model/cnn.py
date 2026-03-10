"""
Two-Headed Residual CNN for Checkers Evaluation
================================================================
This module defines the residual CNN architecture used to predict win
probabilities from a checkers board state.

Architecture overview:
    1. Input projection: Conv2d(5 → hidden_dims) with BatchNorm + ReLU
    2. Residual backbone: N residual blocks, each containing:
       Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm + skip connection → ReLU
    3. Head 1 (p_black): Fully-connected layers → Sigmoid → P(Black wins)
    4. Head 2 (p_white): Fully-connected layers → Sigmoid → P(White wins)

Input encoding (5 channels × 8 rows × 8 cols):
    Channel 0: Black pieces       (1.0 where present, 0.0 elsewhere)
    Channel 1: White pieces       (1.0 where present, 0.0 elsewhere)
    Channel 2: Black kings        (1.0 where present, 0.0 elsewhere)
    Channel 3: White kings        (1.0 where present, 0.0 elsewhere)
    Channel 4: Turn indicator     (all 1.0 if Black's turn, all 0.0 if White's)

Output:
    (p_black, p_white) — independent probabilities in [0, 1] via Sigmoid.
    These are treated as independent predictions, not a probability distribution.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    A single residual block: two 3×3 convolutions with a skip connection.

        input → Conv → BN → ReLU → Conv → BN → (+input) → ReLU → output
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + x)


class TwoHeadedCNN(nn.Module):
    """
    Configurable two-headed residual CNN for board evaluation.

    Args:
        channels:        Number of input channels (default 5 for the board encoding)
        hidden_dims:     Number of filters per conv layer and implicit feature width
        num_conv_layers: Number of residual blocks in the backbone
        dropout_rate:    Dropout probability in the FC head layers (regularization)
    """

    def __init__(self, channels: int = 5, hidden_dims: int = 64,
                 num_conv_layers: int = 2, dropout_rate: float = 0.2):
        super(TwoHeadedCNN, self).__init__()

        # ── Input projection: 5 channels → hidden_dims ──────────────
        self.input_proj = nn.Sequential(
            nn.Conv2d(channels, hidden_dims, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims),
            nn.ReLU(inplace=True),
        )

        # ── Residual backbone ────────────────────────────────────────
        # N residual blocks, each with skip connections
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dims) for _ in range(num_conv_layers)]
        )

        self.flatten = nn.Flatten()

        # After flatten: 8 × 8 × hidden_dims = 64 * hidden_dims features
        flattened_size = 64 * hidden_dims

        # ── Head 1: P(Black wins) ────────────────────────────────────
        self.head_black_win = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # ── Head 2: P(White wins) ────────────────────────────────────
        self.head_white_win = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the network.

        Args:
            x: Tensor of shape (batch_size, 5, 8, 8)

        Returns:
            (p_black, p_white): Tuple of tensors, each shape (batch_size,)
        """
        features = self.input_proj(x)
        features = self.residual_blocks(features)
        features = self.flatten(features)

        p_black = self.head_black_win(features).squeeze()
        p_white = self.head_white_win(features).squeeze()

        return p_black, p_white


def board_to_tensor(grid: list[list[int]], current_turn: int) -> torch.Tensor:
    """
    Convert a Python board grid to the 5-channel tensor expected by TwoHeadedCNN.

    Encoding scheme:
        Channel 0: 1.0 at positions with BLACK pieces (value 1)
        Channel 1: 1.0 at positions with WHITE pieces (value 2)
        Channel 2: 1.0 at positions with BLACK_KING pieces (value 3)
        Channel 3: 1.0 at positions with WHITE_KING pieces (value 4)
        Channel 4: Entirely 1.0 if it's Black's turn, entirely 0.0 if White's

    Args:
        grid:         8×8 list of ints representing the board state
        current_turn: 1 (BLACK) or 2 (WHITE)

    Returns:
        Tensor of shape (5, 8, 8), dtype float32
    """
    tensor = torch.zeros((5, 8, 8), dtype=torch.float32)

    # Channel 4: turn indicator (binary plane)
    if current_turn == 1:  # BLACK's turn
        tensor[4, :, :] = 1.0

    # Channels 0–3: piece positions
    for r in range(8):
        for c in range(8):
            piece = grid[r][c]
            if piece == 1:    # BLACK
                tensor[0, r, c] = 1.0
            elif piece == 2:  # WHITE
                tensor[1, r, c] = 1.0
            elif piece == 3:  # BLACK_KING
                tensor[2, r, c] = 1.0
            elif piece == 4:  # WHITE_KING
                tensor[3, r, c] = 1.0

    return tensor
