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
        (np.int8, (BOARD_VECTOR_DEPTH + 1, BOARD_LENGTH, BOARD_LENGTH)),
    ),
    ("output_best_move_index", np.int32),
    ("output_winner", np.int8),
]


def vectorize_board(board: chess.Board) -> np.ndarray:
    result = np.zeros(
        (BOARD_VECTOR_DEPTH + 1, BOARD_LENGTH, BOARD_LENGTH), dtype=np.int8
    )

    for square, piece in board.piece_map().items():
        index = piece.color * 6 + (piece.piece_type - 1)
        result[index, chess.square_rank(square), chess.square_file(square)] = 1

    # Set turn layer
    result[12] = board.turn == chess.WHITE  # Boolean to int conversion

    return result


def vectorize_game_data(
    game_data: list[tuple[chess.Board, chess.Move, int]], playing_for_whites: bool
) -> np.ndarray:
    result = np.zeros((len(game_data),), dtype=vectorized_game_dtype)

    for i, position in enumerate(game_data):
        board: chess.Board = position[0]
        move: chess.Move = position[1]
        winner: int = position[2]

        result[i][2] = winner

        vectorized_board_from_fen = vectorize_board(board)
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
