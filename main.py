import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import chess.engine
import chess.polyglot
import asyncio
import random

BOARD_LENGTH = 8
BOARD_VECTOR_DEPTH = 12
MOVES_PER_SQUARE = 73
POLICY_OUTPUT_SIZE = BOARD_LENGTH * BOARD_LENGTH * MOVES_PER_SQUARE  # = 4672


class Alpha(nn.Module):
    def __init__(self):
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

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        # Shared convolutional body
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x_conv = F.relu(self.bn4(self.conv4(x)))  # Shared features

        # Policy head forward pass
        x_policy = F.relu(self.policy_bn(self.policy_conv(x_conv)))
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


# Makes a chess engine play against itself, storing position (in FEN) and best move (in UCI algebraic notation)
async def collect_game_data(
    engine_executable: str = "stockfish",
    opening_book_path: str = "book.bin",
    max_opening_moves: int = 12,
) -> list[tuple[str, str]]:
    data = []
    transport, engine = await chess.engine.popen_uci(engine_executable)

    board = chess.Board()
    move_count = 0

    with chess.polyglot.open_reader(opening_book_path) as book:
        while not board.is_game_over():
            if move_count < max_opening_moves:
                try:
                    entry = random.choice(list(book.find_all(board)))
                    move = entry.move
                    print(
                        f"Game {len(data) // 2 + 1}, Move {move_count + 1}: {board.san(move)} (from book)"
                    )
                    data.append((board.fen(), move.uci()))
                    board.push(move)
                except IndexError:
                    result = await engine.play(board, chess.engine.Limit(time=0.1))
                    print(
                        f"Game {len(data) // 2 + 1}, Move {move_count + 1}: {board.san(result.move)} (from Stockfish)"
                    )
                    data.append((board.fen(), result.move.uci()))
                    board.push(result.move)
            else:
                result = await engine.play(board, chess.engine.Limit(time=0.1))
                print(
                    f"Game {len(data) // 2 + 1}, Move {move_count + 1}: {board.san(result.move)} (from Stockfish)"
                )
                data.append((board.fen(), result.move.uci()))
                board.push(result.move)
            move_count += 1

    await engine.quit()
    return data


def main() -> None:
    print("Playing game:")
    _ = asyncio.run(collect_game_data())


if __name__ == "__main__":
    main()
