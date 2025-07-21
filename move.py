import chess
import numpy as np

BOARD_LENGTH = 8

UNDERPROMOTION_MOVE_START = 64


def _encode_underpromotions(move: chess.Move, player_color: chess.Color) -> int:
    from_rank = chess.square_rank(move.from_square)
    from_file = chess.square_file(move.from_square)

    to_rank = chess.square_rank(move.to_square)
    to_file = chess.square_file(move.to_square)

    rank_offset = to_rank - from_rank
    file_offset = to_file - from_file

    # Determine promotion direction relative to the pawn's movement
    # Assuming pawn always moves 1 rank forward for promotion
    if file_offset == 0:
        direction_key = 0  # Straight promotion (e.g., e7e8)
    elif (file_offset == -1 and player_color == chess.WHITE) or (
        file_offset == 1 and player_color == chess.BLACK
    ):
        direction_key = 1  # Diagonal Left (e.g., white d7xc8, black e2xf1)
    elif (file_offset == 1 and player_color == chess.WHITE) or (
        file_offset == -1 and player_color == chess.BLACK
    ):
        direction_key = 2  # Diagonal Right (e.g., white d7xe8, black e2xd1)
    else:
        # This should not happen for valid promotions
        raise ValueError(
            f"Invalid underpromotion offsets: rank_offset={rank_offset}, file_offset={file_offset}"
        )

    base_index = UNDERPROMOTION_MOVE_START  # Which is 64

    # Map piece type and direction to the correct plane index
    # Knight promotions: 64, 65, 66 (straight, diag_L, diag_R)
    # Bishop promotions: 67, 68, 69
    # Rook promotions:   70, 71, 72
    if move.promotion == chess.KNIGHT:
        return base_index + direction_key
    elif move.promotion == chess.BISHOP:
        return base_index + 3 + direction_key
    elif move.promotion == chess.ROOK:
        return base_index + 6 + direction_key
    else:
        # Queen promotions are handled by _encode_queen_like_moves
        # This 'if' condition `move.promotion != chess.QUEEN` ensures this.
        raise ValueError(
            "Queen promotion should not be handled by _encode_underpromotions."
        )


def _encode_knight_moves(rank_offset: int, file_offset: int):
    L_SHAPES = [
        (+2, +1),
        (+2, -1),
        (-2, +1),
        (-2, -1),
        (+1, +2),
        (+1, -2),
        (-1, +2),
        (-1, -2),
    ]

    for i, (l_rank, l_file) in enumerate(L_SHAPES):
        if rank_offset == l_rank and file_offset == l_file:
            return 56 + i


def _encode_queen_like_moves(rank_offset: int, file_offset: int):
    QUEEN_DIRECTIONS = [
        (+1, +0),  # North
        (+1, +1),  # North-East
        (+0, +1),  # East
        (-1, +1),  # South-East
        (-1, +0),  # South
        (-1, -1),  # South-West
        (+0, -1),  # West
        (+1, -1),  # North-West
    ]
    MAX_QUEEN_DISTANCE = 7

    # Determine direction index
    direction_idx = -1
    # Normalize dr, df to get the direction vector (e.g., (1,0) for North)
    norm_dr = 0 if rank_offset == 0 else rank_offset // abs(rank_offset)
    norm_df = 0 if file_offset == 0 else file_offset // abs(file_offset)

    for i, (dir_r, dir_f) in enumerate(QUEEN_DIRECTIONS):
        if norm_dr == dir_r and norm_df == dir_f:
            direction_idx = i
            break

    # Determine distance
    # For straight moves (rank_offset=0 or file_offset=0), distance is abs(rank_offset) or abs(file_offset)
    # For diagonal moves, distance is abs(rank_offset) (which is equal to abs(file_offset))
    distance = max(abs(rank_offset), abs(file_offset))

    # Queen-style layers are 0-55
    # Layer = (direction_idx * MAX_QUEEN_DISTANCE) + (distance - 1)
    # Example: Direction N (idx 0), distance 1 -> layer 0
    #          Direction N (idx 0), distance 7 -> layer 6
    #          Direction NE (idx 1), distance 1 -> layer 7
    #          Direction NW (idx 7), distance 7 -> layer 7*7 + 6 = 55
    return (direction_idx * MAX_QUEEN_DISTANCE) + (distance - 1)


def _precomputate_non_underpromotion() -> np.ndarray:
    result = np.full((64, 64), -1, dtype=np.int8)

    for from_square in chess.SQUARES:
        from_rank = chess.square_rank(from_square)
        from_file = chess.square_file(from_square)

        for to_square in chess.SQUARES:
            if from_square == to_square:
                continue

            to_rank = chess.square_rank(to_square)
            to_file = chess.square_file(to_square)

            rank_offset = to_rank - from_rank
            file_offset = to_file - from_file

            if (abs(rank_offset) == 1 and abs(file_offset) == 2) or (
                abs(rank_offset) == 2 and abs(file_offset) == 1
            ):
                result[from_square, to_square] = _encode_knight_moves(
                    rank_offset, file_offset
                )

            # Straight or diagonal line
            if (
                rank_offset == 0
                or file_offset == 0
                or abs(rank_offset) == abs(file_offset)
            ):
                result[from_square, to_square] = _encode_queen_like_moves(
                    rank_offset, file_offset
                )

    return result


NON_UNDERPROMOTIONS = _precomputate_non_underpromotion()


# Assumes move is legal
def encode_move_layer(move: chess.Move, player_color: chess.Color) -> int:
    if move.promotion is not None and move.promotion != chess.QUEEN:
        return _encode_underpromotions(move, player_color)

    return NON_UNDERPROMOTIONS[move.from_square, move.to_square]
