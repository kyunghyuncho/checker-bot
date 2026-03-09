import torch
import torch.nn as nn

class TwoHeadedCNN(nn.Module):
    """
    A "Two-Headed" Convolutional Neural Network with configurable depth.
    
    Input: An 8x8x5 tensor. The 5 channels represent:
      0: Black pieces (1 if black, 0 otherwise)
      1: White pieces (1 if white, 0 otherwise)
      2: Black Kings (1 if black king, 0 otherwise)
      3: White Kings (1 if white king, 0 otherwise)
      4: Turn indicator (All 1s if Black's turn, all 0s if White's turn)
    """
    def __init__(self, channels: int = 5, hidden_dims: int = 64, 
                 num_conv_layers: int = 2, dropout_rate: float = 0.2):
        super(TwoHeadedCNN, self).__init__()
        
        # Dynamically build the shared convolutional backbone
        layers = []
        in_ch = channels
        for _ in range(num_conv_layers):
            layers.extend([
                nn.Conv2d(in_channels=in_ch, out_channels=hidden_dims, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dims),
                nn.ReLU(),
            ])
            in_ch = hidden_dims
        layers.append(nn.Flatten())
        self.shared_backbone = nn.Sequential(*layers)
        
        # 8*8 = 64 spatial locations * hidden_dims channels
        flattened_size = 64 * hidden_dims
        
        # Head 1: Probability of Black Winning (P_black)
        self.head_black_win = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Head 2: Probability of White Winning (P_white)
        self.head_white_win = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        Expects x of shape (Batch_Size, 5, 8, 8).
        Returns a tuple: (p_black, p_white)
        """
        features = self.shared_backbone(x)
        
        p_black = self.head_black_win(features).squeeze()
        p_white = self.head_white_win(features).squeeze()
        
        return p_black, p_white

def board_to_tensor(grid: list[list[int]], current_turn: int) -> torch.Tensor:
    """
    Converts a Python 2D list Checkers board to the 5-channel PyTorch tensor.
    Handles the encoding logic required by TwoHeadedCNN.
    """
    # Initialize an empty tensor of shape (5 channels, 8 rows, 8 cols)
    tensor = torch.zeros((5, 8, 8), dtype=torch.float32)
    
    # Fill the turn channel. If Black (1), entirely 1s. If White (2), 0s.
    if current_turn == 1:
        tensor[4, :, :] = 1.0
        
    for r in range(8):
        for c in range(8):
            piece = grid[r][c]
            if piece == 1: # BLACK
                tensor[0, r, c] = 1.0
            elif piece == 2: # WHITE
                tensor[1, r, c] = 1.0
            elif piece == 3: # BLACK_KING
                tensor[2, r, c] = 1.0
            elif piece == 4: # WHITE_KING
                tensor[3, r, c] = 1.0
                
    return tensor
