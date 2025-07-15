import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import chess.engine
import chess.polyglot
from decode_move import encode_move_layer

BOARD_LENGTH = 8
BOARD_VECTOR_DEPTH = 13  # 6x2+1
MOVES_PER_SQUARE = 73
POLICY_OUTPUT_SIZE = BOARD_LENGTH * BOARD_LENGTH * MOVES_PER_SQUARE  # = 4672

WIN = 1
DRAW = 0
LOSE = -1


class Alpha(nn.Module):
    """
    AlphaZero-like neural network architecture for chess.

    Input: A 13x8x8 tensor, 12 because there are 6 pieces in chess, and 2 colors. The 13th layer is for the turn, all 1 if whites, all 0 if black.
    Output: AlphaZero move representation, can see decode_move's logic to understand more

    This network takes a board representation as input and outputs two heads:
    - A policy head: predicts a probability distribution over possible moves.
    - A value head: estimates the winning probability for the current player.
    """

    def __init__(self):
        """
        Initializes the Alpha model's layers.
        """
        super(Alpha, self).__init__()

        # Shared convolutional body
        self.conv1 = nn.Conv2d(BOARD_VECTOR_DEPTH, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        # Policy head: predicts a 73-channel move distribution for each square
        self.policy_conv = nn.Conv2d(256, MOVES_PER_SQUARE, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(MOVES_PER_SQUARE)

        # Value head: evaluates board position as a scalar in [-1, 1]
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(BOARD_LENGTH * BOARD_LENGTH, 256)
        self.value_fc2 = nn.Linear(256, 1)
        self.value_dr1 = nn.Dropout()
        self.value_dr2 = nn.Dropout()

    def forward(
        self, legality_mask: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the network.

        Args:
            legality_mask (torch.Tensor): A tensor representing the legality of moves,
                                          used to mask out illegal moves in the policy head.
            x (torch.Tensor): The input tensor representing the chess board state.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - x_policy (torch.Tensor): The log-softmax probabilities for each move.
                - x_value (torch.Tensor): The predicted value of the board position,
                                          scaled to [-1, 1].
        """
        # Shared convolutional body
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x_conv = F.relu(self.bn4(self.conv4(x)))  # Shared features

        # Policy head forward pass
        x_policy = F.relu(self.policy_bn(self.policy_conv(x_conv)))
        x_policy += legality_mask
        x_policy = x_policy.view(-1, POLICY_OUTPUT_SIZE)
        x_policy = F.log_softmax(x_policy, dim=1)  # Log-probabilities for NLLLoss

        # Value head forward pass
        x_value = F.relu(self.value_bn(self.value_conv(x_conv)))

        # Flatten spatial dimensions
        x_value = x_value.view(-1, BOARD_LENGTH * BOARD_LENGTH)
        x_value = F.relu(self.value_fc1(self.value_dr1(x_value)))

        # Output ∈ [-1, 1]
        x_value = torch.tanh(self.value_fc2(self.value_dr2(x_value)))

        return x_policy, x_value


def generate_legality_mask(board: chess.Board) -> torch.Tensor:
    """
    Generates a legality mask for the given chess board.

    The mask is a tensor of shape (MOVES_PER_SQUARE, BOARD_LENGTH, BOARD_LENGTH),
    where illegal moves are set to a very small negative number (-1e9) and legal
    moves are set to 0.0. This mask can be added to the policy head output
    to effectively zero out the probabilities of illegal moves after a softmax.

    Args:
        board (chess.Board): The current chess board state.

    Returns:
        torch.Tensor: A tensor representing the legality mask.
    """
    mask = torch.full((MOVES_PER_SQUARE, BOARD_LENGTH, BOARD_LENGTH), -1e9)

    for move in board.legal_moves:
        from_rank = move.from_square // BOARD_LENGTH
        from_file = move.from_square % BOARD_LENGTH

        mask[encode_move_layer(move)][from_rank][from_file] = 0.0

    return mask


def main() -> None:
    print("Hello world!")


if __name__ == "__main__":
    main()
