import chess
import chess.engine
import chess.polyglot
import chess.pgn
import numpy as np
from typing import TextIO, Iterator
from move import encode_move_layer


WIN = 1
DRAW = 0
LOSE = -1

BOARD_LENGTH = 8
BOARD_VECTOR_DEPTH = 12  # 6x2

vectorized_game_dtype = [
    (
        "input_board",
        (np.int8, (BOARD_VECTOR_DEPTH + 1, BOARD_LENGTH, BOARD_LENGTH)),
    ),
    ("output_best_move_index", np.int32),
    ("output_win", np.int8),
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


def get_win(result: str, turn: chess.Color) -> int:
    match result:
        case "1/2-1/2":
            return DRAW
        case "1-0":
            return WIN if turn == chess.WHITE else LOSE
        case "0-1":
            return WIN if turn == chess.BLACK else LOSE


def vectorize_game_data(
    game_data: list[tuple[chess.Board, chess.Move, str]],
) -> np.ndarray:
    vectorized = np.zeros(len(game_data), dtype=vectorized_game_dtype)

    for i, (board, move, outcome) in enumerate(game_data):
        vectorized["output_win"][i] = get_win(outcome, board.turn)
        vectorized["input_board"][i] = vectorize_board(board)

        move_from_rank = chess.square_rank(move.from_square)
        move_from_file = chess.square_file(move.from_square)
        move_depth = encode_move_layer(move, board.turn)

        move_index = (
            move_depth * BOARD_LENGTH * BOARD_LENGTH
            + move_from_rank * BOARD_LENGTH
            + move_from_file
        )

        vectorized["output_best_move_index"][i] = move_index

    return vectorized


def generate_game_chunks(file_handler: TextIO, chunk_size: int) -> Iterator[np.ndarray]:
    current_chunk = []

    while True:
        game = chess.pgn.read_game(file_handler)

        if game is None:
            if current_chunk:  # Yield any remaining games in the last chunk
                yield np.array(current_chunk, dtype=object)
            break

        current_chunk.append(game)

        if len(current_chunk) == chunk_size:
            yield np.array(current_chunk, dtype=object)
            current_chunk = []
