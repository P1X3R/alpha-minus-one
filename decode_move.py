QUEEN_MOVE_START = 0
QUEEN_MOVE_END = 55
KNIGHT_MOVE_START = 56
KNIGHT_MOVE_END = 63
UNDERPROMOTION_MOVE_START = 64
UNDERPROMOTION_MOVE_END = 72


def _parse_queen_style_move(
    move_layer: int, from_rank: int, from_file: int
) -> tuple[int, int]:
    """
    Parses a 'queen-style' move layer (0-55) to determine the destination square.
    """
    to_rank: int = from_rank
    to_file: int = from_file
    distance: int

    directions = [
        (0, 6, 1, 0),  # top (dr=1, df=0)
        (7, 13, 1, 1),  # top-right (dr=1, df=1)
        (14, 20, 0, 1),  # right (dr=0, df=1)
        (21, 27, -1, 1),  # bottom-right (dr=-1, df=1)
        (28, 34, -1, 0),  # bottom (dr=-1, df=0)
        (35, 41, -1, -1),  # bottom-left (dr=-1, df=-1)
        (42, 48, 0, -1),  # left (dr=0, df=-1)
        (49, 55, 1, -1),  # top-left (dr=1, df=-1)
    ]

    for start, end, direction_rank, direction_file in directions:
        if move_layer >= start and move_layer <= end:
            distance = (move_layer - start) + 1
            to_rank = from_rank + distance * direction_rank
            to_file = from_file + distance * direction_file
            return to_rank, to_file

    raise ValueError(f"Invalid move_layer for queen-style move: {move_layer}")

    return to_rank, to_file


def _parse_knight_move(
    move_layer: int, from_rank: int, from_file: int
) -> tuple[int, int]:
    """
    Parses a knight move layer (56-63) to determine the destination square.
    """
    to_rank: int = from_rank
    to_file: int = from_file

    # Knight moves (56-63)
    if move_layer == 56:  # +2, -1 (top-left)
        to_rank += 2
        to_file -= 1
    elif move_layer == 57:  # +2, +1 (top-right)
        to_rank += 2
        to_file += 1
    elif move_layer == 58:  # +1, +2 (right-top)
        to_rank += 1
        to_file += 2
    elif move_layer == 59:  # -1, +2 (right-bottom)
        to_rank -= 1
        to_file += 2
    elif move_layer == 60:  # -2, +1 (bottom-right)
        to_rank -= 2
        to_file += 1
    elif move_layer == 61:  # -2, -1 (bottom-left)
        to_rank -= 2
        to_file -= 1
    elif move_layer == 62:  # -1, -2 (left-bottom)
        to_rank -= 1
        to_file -= 2
    elif move_layer == 63:  # +1, -2 (left-top)
        to_rank += 1
        to_file -= 2
    else:
        raise ValueError(f"Invalid move_layer for knight move: {move_layer}")

    return to_rank, to_file


def _parse_underpromotion_move(
    move_layer: int, from_rank: int, from_file: int
) -> tuple[int, int, str]:
    """
    Parses an underpromotion move layer (64-72) to determine the destination square
    and the promoted piece type.
    """
    to_file: int = from_file
    promotion_piece: str

    # Determine the destination rank (always 8th rank for white, 1st for black)
    if from_rank == 6:  # White pawn on 7th rank
        to_rank = 7  # Promotes to 8th rank
    elif from_rank == 1:  # Black pawn on 2nd rank
        to_rank = 0  # Promotes to 1st rank
    else:
        # This indicates an invalid call for an underpromotion move
        raise ValueError(
            f"Invalid from_rank {from_rank} for underpromotion move_layer {move_layer}. "
            "Pawn must be on 2nd or 7th rank."
        )

    relative_layer = move_layer - UNDERPROMOTION_MOVE_START  # Gives a value from 0 to 8

    # Promote to Knight (layers 0-2 relative to UNDERPROMOTION_MOVE_START)
    if relative_layer >= 0 and relative_layer <= 2:
        promotion_piece = "n"  # Knight
        if relative_layer == 0:  # Straight promotion
            to_file = from_file
        elif relative_layer == 1:  # Diagonal-left promotion
            to_file = from_file - 1
        elif relative_layer == 2:  # Diagonal-right promotion
            to_file = from_file + 1

    # Promote to Bishop (layers 3-5 relative)
    elif relative_layer >= 3 and relative_layer <= 5:
        promotion_piece = "b"  # Bishop
        if relative_layer == 3:  # Straight promotion
            to_file = from_file
        elif relative_layer == 4:  # Diagonal-left promotion
            to_file = from_file - 1
        elif relative_layer == 5:  # Diagonal-right promotion
            to_file = from_file + 1

    # Promote to Rook (layers 6-8 relative)
    elif relative_layer >= 6 and relative_layer <= 8:
        promotion_piece = "r"  # Rook
        if relative_layer == 6:  # Straight promotion
            to_file = from_file
        elif relative_layer == 7:  # Diagonal-left promotion
            to_file = from_file - 1
        elif relative_layer == 8:  # Diagonal-right promotion
            to_file = from_file + 1
    else:
        raise ValueError(f"Invalid relative_layer for underpromotion: {relative_layer}")

    return to_rank, to_file, promotion_piece


def decode_move(
    move_layer: int, from_rank: int, from_file: int
) -> tuple[int, int, str | None]:
    """
    Decodes an AlphaZero policy head move layer into a destination square
    and optional promotion piece.

    Args:
        move_layer: An integer from 0 to 72 representing the move type.
        from_rank: The 0-indexed rank (row) of the starting square.
        from_file: The 0-indexed file (column) of the starting square.

    Returns:
        A tuple (to_rank, to_file, promotion_piece).
        - to_rank: The 0-indexed rank of the destination square.
        - to_file: The 0-indexed file of the destination square.
        - promotion_piece: 'N', 'B', 'R' for underpromotions, otherwise None.
    """

    if QUEEN_MOVE_START <= move_layer <= QUEEN_MOVE_END:
        to_rank, to_file = _parse_queen_style_move(move_layer, from_rank, from_file)
        return to_rank, to_file, None
    elif KNIGHT_MOVE_START <= move_layer <= KNIGHT_MOVE_END:
        to_rank, to_file = _parse_knight_move(move_layer, from_rank, from_file)
        return to_rank, to_file, None
    elif UNDERPROMOTION_MOVE_START <= move_layer <= UNDERPROMOTION_MOVE_END:
        to_rank, to_file, promotion_piece = _parse_underpromotion_move(
            move_layer, from_rank, from_file
        )
        return to_rank, to_file, promotion_piece
    else:
        raise ValueError(f"Invalid move_layer: {move_layer}. Must be between 0 and 72.")
