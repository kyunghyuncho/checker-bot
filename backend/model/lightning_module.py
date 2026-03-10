"""
PyTorch Lightning Training Module for Checkers
================================================
Wraps the TwoHeadedCNN in a Lightning module for clean training loops,
automatic logging, and checkpoint management.

Training data flow:
    1. CheckersDataset loads game histories from a JSON file
    2. Each board state is converted to a 5-channel tensor via board_to_tensor()
    3. Labels are derived from the game outcome:
       - Black win → target = [1.0, 0.0]
       - White win → target = [0.0, 1.0]
       - Draw      → target = [0.5, 0.5]
    4. The model predicts (p_black, p_white), both in [0, 1]
    5. Loss = BCE(p_black, target_black) + BCE(p_white, target_white)

Note: every state in a game's history shares the same label (the game's outcome).
This is "outcome-based" supervision — every position in a winning game is labeled
as a winning position, regardless of the specific board state.
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import json

from backend.model.cnn import TwoHeadedCNN, board_to_tensor


class CheckersDataset(Dataset):
    """
    In-memory dataset of labeled checkers board states.

    Reads from a JSON file produced by generator.py. Each game contains a
    history of board states, all labeled with the game's final outcome.

    Exponential label discounting (γ) is applied at load time:
        discounted_label = 0.5 + (raw_label − 0.5) × (1 − γ)^d
    where d = distance from the final position in that game.
    γ=0: no discounting (original behavior)
    γ=1: only the last position gets the true label

    Each sample is a (tensor, target) pair:
        - tensor: shape (5, 8, 8) — board encoding for TwoHeadedCNN
        - target: shape (2,) — [P(black_wins), P(white_wins)]
    """

    def __init__(self, json_file: str, discount_factor: float = 0.0):
        """Load all board states from the dataset JSON file into memory."""
        with open(json_file, 'r') as f:
            data = json.load(f)

        self.samples = []
        for game in data:
            history = game.get('history', [])
            if not history:
                continue

            # Raw outcome label (same for all states in this game)
            raw_label = history[0].get('label', 0.5)
            T = len(history)

            for t, state in enumerate(history):
                # Convert the raw board grid to a 5-channel tensor
                tensor = board_to_tensor(state['board'], state['turn'])

                # Apply exponential discounting toward 0.5
                d = T - 1 - t  # distance from end
                discount = (1.0 - discount_factor) ** d
                label = 0.5 + (raw_label - 0.5) * discount

                # Derive targets for both heads
                target_black = float(label)
                target_white = 1.0 - float(label)
                if abs(target_black - 0.5) < 1e-6:
                    target_white = 0.5  # Draw: both heads predict 0.5

                target = torch.tensor([target_black, target_white], dtype=torch.float32)
                self.samples.append((tensor, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class CheckersLightningModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for training the TwoHeadedCNN.

    Handles:
        - Forward pass delegation to TwoHeadedCNN
        - Binary Cross-Entropy loss computation for both prediction heads
        - Metric logging (per-head loss and combined loss)
        - Adam optimizer configuration

    All hyperparameters are saved via save_hyperparameters() so they are
    automatically stored in checkpoints and can be restored on load.
    """

    def __init__(self, learning_rate: float = 1e-3, channels: int = 5,
                 hidden_dims: int = 64, num_conv_layers: int = 2,
                 dropout_rate: float = 0.2):
        super().__init__()
        self.save_hyperparameters()  # Stores all args for checkpoint reproducibility

        # Instantiate the CNN with the provided architecture hyperparameters
        self.model = TwoHeadedCNN(
            channels=channels,
            hidden_dims=hidden_dims,
            num_conv_layers=num_conv_layers,
            dropout_rate=dropout_rate
        )
        self.learning_rate = learning_rate

    def forward(self, x):
        """Delegate to the underlying TwoHeadedCNN."""
        return self.model(x)

    def _shared_step(self, batch, batch_idx, phase: str):
        """
        Compute loss for a batch (used by both training and validation steps).

        Args:
            batch:     (x, y) where x is (B, 5, 8, 8) and y is (B, 2)
            batch_idx: Index of the batch (unused, required by Lightning)
            phase:     "train" or "val" — used as prefix for logged metric names

        Returns:
            Combined loss (scalar tensor)
        """
        x, y = batch

        # Forward pass: get independent win probabilities
        p_black, p_white = self(x)

        # Split target into per-head targets
        target_black = y[:, 0]  # P(Black wins)
        target_white = y[:, 1]  # P(White wins)

        # Binary Cross-Entropy for each head independently
        loss_black = F.binary_cross_entropy(p_black, target_black)
        loss_white = F.binary_cross_entropy(p_white, target_white)

        # Combined loss: equally weight both heads
        loss = loss_black + loss_white

        # Log metrics for the loss chart and early stopping
        self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{phase}_loss_black", loss_black, on_step=False, on_epoch=True)
        self.log(f"{phase}_loss_white", loss_white, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        """Lightning training step — delegates to shared loss computation."""
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        """Lightning validation step — delegates to shared loss computation."""
        return self._shared_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        """
        Configure the Adam optimizer.
        
        Adam is chosen for its robustness with small datasets and shallow networks.
        Learning rate is configurable via the constructor.
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
