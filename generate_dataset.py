import chess
import chess.engine
import chess.polyglot
import random


async def collect_game_data(
    engine_executable: str = "stockfish",
    opening_book_path: str = "book.bin",
    max_opening_moves: int = 12,
    color_learning: chess.Color = chess.WHITE,
) -> list[tuple[str, str, int]]:
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
        list[tuple[str, str]]: A list of (FEN, UCI move) tuples representing the game.
    """
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
                    data.append((board.fen(), move.uci(), DRAW))
                    board.push(move)
                except IndexError:  # If no moves found in book, use engine
                    result = await engine.play(board, chess.engine.Limit(time=0.1))
                    print(
                        f"Game {len(data) // 2 + 1}, Move {move_count + 1}: {board.san(result.move)} (from Stockfish)"
                    )
                    data.append((board.fen(), result.move.uci(), DRAW))
                    board.push(result.move)
            else:
                result = await engine.play(board, chess.engine.Limit(time=0.1))
                print(
                    f"Game {len(data) // 2 + 1}, Move {move_count + 1}: {board.san(result.move)} (from Stockfish)"
                )
                data.append((board.fen(), result.move.uci(), DRAW))
                board.push(result.move)
            move_count += 1

    winner = board.outcome().winner

    if winner == color_learning:
        data = [(position[0], position[1], WIN) for position in data]
    elif winner is not None:
        data = [(position[0], position[1], LOSE) for position in data]

    await engine.quit()
    return data


def vectorize_fen(fen: str) -> np.ndarray:
    """
    Converts a FEN (Forsyth-Edwards Notation) string into a 12x8x8 tensor
    representing the chess board state. Each of the 12 channels corresponds
    to a specific piece type (e.g., white pawns, black rooks).

    Args:
        fen (str): The FEN string of the chess board.

    Returns:
        torch.Tensor: A tensor of shape (12, 8, 8) representing the board,
                      one-hot encoded for each piece type.
    """
    result = np.zeros((BOARD_VECTOR_DEPTH, BOARD_LENGTH, BOARD_LENGTH))
    rank = BOARD_LENGTH - 1
    file = 0

    for char in fen[: fen.find(" ")]:
        if char == "/":
            rank -= 1
            file = 0
            continue
        elif char.isdigit():
            file += int(char)
            continue

        match char:
            case "P":
                result[0][rank][file] = 1
            case "N":
                result[1][rank][file] = 1
            case "B":
                result[2][rank][file] = 1
            case "R":
                result[3][rank][file] = 1
            case "Q":
                result[4][rank][file] = 1
            case "K":
                result[5][rank][file] = 1
            case "p":
                result[6][rank][file] = 1
            case "n":
                result[7][rank][file] = 1
            case "b":
                result[8][rank][file] = 1
            case "r":
                result[9][rank][file] = 1
            case "q":
                result[10][rank][file] = 1
            case "k":
                result[11][rank][file] = 1

        file += 1

    return result
