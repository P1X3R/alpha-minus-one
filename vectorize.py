import chess
import chess.engine
import chess.polyglot
import chess.pgn
import numpy as np
import h5py
from itertools import batched
from typing import TextIO, Iterator
from move import encode_move_layer
from multiprocessing import Pool


WIN = 1
DRAW = 0
LOSE = -1

BOARD_LENGTH = 8
BOARD_VECTOR_DEPTH = 12  # 6x2

vectorized_position_dtype = [
    (
        "input_board",
        (np.int8, (BOARD_VECTOR_DEPTH + 1, BOARD_LENGTH, BOARD_LENGTH)),
    ),
    ("output_best_move_index", np.int16),
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


def vectorize_game(
    game: chess.pgn.Game,
) -> np.ndarray:
    mainline = list(game.mainline())
    mainline.pop(0)

    vectorized = np.zeros(len(mainline), dtype=vectorized_position_dtype)
    game_result = game.headers.get("Result")

    for i, position in enumerate(mainline):
        board = position.board()
        move = position.move

        vectorized["output_win"][i] = get_win(game_result, board.turn)
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


def read_game_chunks(
    file_handler: TextIO, chunk_size: int
) -> Iterator[list[chess.pgn.Game]]:
    current_chunk = []

    while True:
        game = chess.pgn.read_game(file_handler)

        # Yield any remaining games in the last chunk
        if game is None and current_chunk:
            yield current_chunk
            break
        if game.headers.get("Result") == "*":
            continue

        current_chunk.append(game)

        if len(current_chunk) == chunk_size:
            yield current_chunk
            current_chunk = []


def vectorize_dataset_part(
    dataset_path: str,
    output_vectorized_path: str,
    pgn_chunk_size: int,
    vectorization_chunk_size: int,
):
    if vectorization_chunk_size > pgn_chunk_size:
        vectorization_chunk_size = pgn_chunk_size

    with open(dataset_path, "r") as input_file:
        with h5py.File(output_vectorized_path, "w") as output:
            dataset = output.create_dataset(
                "games",
                shape=(0,),
                maxshape=(None,),
                compression="lzf",
                shuffle=True,
                chunks=True,
                dtype=vectorized_position_dtype,
            )

            for chunk in read_game_chunks(input_file, pgn_chunk_size):
                for slize in batched(chunk, vectorization_chunk_size):
                    vectorized: np.ndarray = np.concatenate(
                        list(map(vectorize_game, slize))
                    )

                    # Append new vectorized data
                    current_size = dataset.shape[0]
                    dataset.resize(current_size + len(vectorized), axis=0)
                    dataset[current_size:] = vectorized


def vectorize_dataset(
    dataset_paths: list[str],
    output_vectorized_path: list[str],
    pgn_chunk_size: int,
    vectorization_chunk_size: int,
):
    with Pool() as pool:
        args = zip(
            dataset_paths,
            output_vectorized_path,
            [pgn_chunk_size] * len(dataset_paths),
            [vectorization_chunk_size] * len(dataset_paths),
        )

        pool.starmap(vectorize_dataset_part, args)
