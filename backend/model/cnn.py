"""
Two-Headed Convolutional Neural Network for Checkers Evaluation
================================================================
This module defines the CNN architecture used to predict win probabilities
from a checkers board state.

Architecture overview:
    1. Shared convolutional backbone: N conv layers (3×3, same-padding) with
       BatchNorm and ReLU. Processes the 5-channel input into spatial features.
    2. Head 1 (p_black): Fully-connected layers → Sigmoid → P(Black wins)
    3. Head 2 (p_white): Fully-connected layers → Sigmoid → P(White wins)

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


class TwoHeadedCNN(nn.Module):
    """
    Configurable two-headed CNN for board evaluation.

    Args:
        channels:        Number of input channels (default 5 for the board encoding)
        hidden_dims:     Number of filters per conv layer and implicit feature width
        num_conv_layers: Number of stacked Conv2d → BatchNorm → ReLU blocks
        dropout_rate:    Dropout probability in the FC head layers (regularization)
    """

    def __init__(self, channels: int = 5, hidden_dims: int = 64,
                 num_conv_layers: int = 2, dropout_rate: float = 0.2):
        super(TwoHeadedCNN, self).__init__()

        # ── Shared convolutional backbone ────────────────────────────
        # Each layer: Conv2d(3×3, same padding) → BatchNorm → ReLU
        # Same padding preserves the 8×8 spatial dimensions throughout
        layers = []
        in_ch = channels
        for _ in range(num_conv_layers):
            layers.extend([
                nn.Conv2d(in_channels=in_ch, out_channels=hidden_dims,
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dims),
                nn.ReLU(),
            ])
            in_ch = hidden_dims  # All subsequent layers use hidden_dims as input
        layers.append(nn.Flatten())  # Flatten 8×8×hidden_dims → single vector
        self.shared_backbone = nn.Sequential(*layers)

        # After flatten: 8 × 8 × hidden_dims = 64 * hidden_dims features
        flattened_size = 64 * hidden_dims

        # ── Head 1: P(Black wins) ────────────────────────────────────
        # Linear → ReLU → Dropout → Linear → Sigmoid
        self.head_black_win = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output in [0, 1]
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
        features = self.shared_backbone(x)

        p_black = self.head_black_win(features).squeeze()  # (batch,)
        p_white = self.head_white_win(features).squeeze()  # (batch,)

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
