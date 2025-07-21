import chess
import chess.engine
import chess.polyglot
import random
import h5py
import asyncio
from vectorize import vectorize_game_data, vectorized_game_dtype


DATASET_PATH = "data.hdf5"

WIN = 1
DRAW = 0
LOSE = -1

BOARD_LENGTH = 8
BOARD_VECTOR_DEPTH = 12  # 6x2


async def collect_game_data(
    engine_executable: str = "stockfish",
    opening_book_path: str = "book.bin",
    max_opening_moves: int = 12,
    color_learning: chess.Color = chess.WHITE,
    threads: int = 1,
) -> list[tuple[str, chess.Move, int]]:
    """
    Makes a chess engine play against itself, collecting game data.

    The game data consists of a list of tuples, where each tuple contains
    the FEN (Forsyth-Edwards Notation) string of a position and the UCI
    (Universal Chess Interface) algebraic notation of the best move played
    from that position. The function uses an opening book for the initial
    moves and then switches to the specified chess engine.

    Args:
        engine_executable (str): The path to the chess engine executable (e.g., "stockfish").
        opening_book_path (str): The path to the Polyglot opening book file.
        max_opening_moves (int): The maximum number of moves to play from the opening book.

    Returns:
        list[tuple[str, chess.Move, int]]: A list of (FEN, move, winner) tuples representing the game.
    """
    data = []
    transport, engine = await chess.engine.popen_uci(engine_executable)
    await engine.configure({"Threads": threads})

    board = chess.Board()
    move_count = 0

    ENGINE_TIME_LIMIT = 0.5

    with chess.polyglot.open_reader(opening_book_path) as book:
        while not board.is_game_over():
            if move_count < max_opening_moves:
                try:
                    entry = random.choice(list(book.find_all(board)))
                    move = entry.move
                    print(
                        f"Game {len(data) // 2 + 1}, Move {move_count + 1}: {board.san(move)} (from book)"
                    )
                    data.append((board.fen(), move, DRAW))
                    board.push(move)
                except IndexError:  # If no moves found in book, use engine
                    result = await engine.play(
                        board, chess.engine.Limit(time=ENGINE_TIME_LIMIT)
                    )
                    print(
                        f"Game {len(data) // 2 + 1}, Move {move_count + 1}: {board.san(result.move)} (from Stockfish)"
                    )
                    data.append((board.fen(), result.move, DRAW))
                    board.push(result.move)
            else:
                result = await engine.play(
                    board, chess.engine.Limit(time=ENGINE_TIME_LIMIT)
                )
                print(
                    f"Game {len(data) // 2 + 1}, Move {move_count + 1}: {board.san(result.move)} (from Stockfish)"
                )
                data.append((board.fen(), result.move, DRAW))
                board.push(result.move)
            move_count += 1

    winner = board.outcome().winner

    if winner == color_learning:
        data = [(position[0], position[1], WIN) for position in data]
    elif winner is not None:
        data = [(position[0], position[1], LOSE) for position in data]

    await engine.quit()
    return data


def generate_dataset():
    with h5py.File(name=DATASET_PATH, mode="w") as file:
        dataset = file.create_dataset(
            "games",
            shape=(0,),
            maxshape=(None,),
            dtype=vectorized_game_dtype,
            shuffle=True,
        )

        games: int = 0

        while True:
            print("-----------------")
            print("--- New game! ---")
            print("-----------------")

            playing_for = random.choice([True, False])

            no_proccesed_data = (
                asyncio.run(collect_game_data(color_learning=playing_for)),
            )

            game = vectorize_game_data(
                no_proccesed_data[0],
                playing_for,
            )

            current_rows = dataset.shape[0]
            dataset.resize(current_rows + game.shape[0], axis=0)

            games += 1

            dataset[current_rows:] = game

            print(f"\nTotal games: {games}")
