import chess
import chess.engine
import chess.polyglot
import numpy as np
from move import encode_move_layer


BOARD_LENGTH = 8
BOARD_VECTOR_DEPTH = 12  # 6x2

vectorized_game_dtype = [
    (
        "input_board",
        (np.int32, (BOARD_VECTOR_DEPTH + 1, BOARD_LENGTH, BOARD_LENGTH)),
    ),
    ("output_best_move_index", np.int32),
    ("output_winner", np.int32),
]


def vectorize_fen(fen: str) -> np.ndarray:
    """
    Converts a FEN (Forsyth-Edwards Notation) string into a 12x8x8 tensor
    representing the chess board state. Each of the 12 channels corresponds
    to a specific piece type (e.g., white pawns, black rooks).

    Args:
        fen (str): The FEN string of the chess board.

    Returns:
        np.ndarray: An integer tensor of shape (12, 8, 8) representing the board,
                      one-hot encoded for each piece type.
    """
    result = np.zeros((BOARD_VECTOR_DEPTH, BOARD_LENGTH, BOARD_LENGTH), dtype=int)
    rank = BOARD_LENGTH - 1
    file = 0

    for char in fen.split(" ")[0]:
        if char == "/":
            rank -= 1
            file = 0
            continue
        elif char.isdigit():
            file += int(char)
            continue

        for i, type_char in enumerate("PNBRQKpnbrqk"):
            if char == type_char:
                result[i, rank, file] = 1
                break

        file += 1

    return result


def vectorize_game_data(
    game_data: list[tuple[str, chess.Move, int]], playing_for_whites: bool
) -> np.ndarray:
    result = np.zeros((len(game_data),), dtype=vectorized_game_dtype)

    for i, position in enumerate(game_data):
        fen: str = position[0]
        move: chess.Move = position[1]
        winner: int = position[2]

        result[i][2] = winner

        vectorized_board_from_fen = vectorize_fen(fen)
        turn_layer = np.full(
            (1, BOARD_LENGTH, BOARD_LENGTH),
            1 if playing_for_whites else 0,
            dtype=np.int32,
        )
        vectorized_board = np.concatenate(
            (vectorized_board_from_fen, turn_layer), axis=0
        )

        result[i][0] = vectorized_board

        move_from_rank = move.from_square // BOARD_LENGTH
        move_from_file = move.from_square % BOARD_LENGTH
        move_depth = encode_move_layer(move, chess.WHITE if i % 2 != 0 else chess.BLACK)

        move_index = (
            move_depth * BOARD_LENGTH * BOARD_LENGTH
            + move_from_rank * BOARD_LENGTH
            + move_from_file
        )

        result[i][1] = move_index

    return result
