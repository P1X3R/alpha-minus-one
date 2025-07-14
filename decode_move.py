import chess

BOARD_LENGTH = 8

# Define the start and end indices for different move types within the 73-layer policy head.
# These constants help in mapping a move's properties to a specific layer in the neural network's
# policy output.
QUEEN_MOVE_START = 0
QUEEN_MOVE_END = 55  # Represents 8 directions * 7 possible distances = 56 layers (0-55)
KNIGHT_MOVE_START = 56
KNIGHT_MOVE_END = 63  # Represents the 8 possible knight moves (56-63)
UNDERPROMOTION_MOVE_START = 64
UNDERPROMOTION_MOVE_END = 72  # Represents 3 pieces (N, B, R) * 3 directions (straight, diag-left, diag-right) = 9 layers (64-72)


def _parse_queen_style_move(
    move_layer: int, from_rank: int, from_file: int
) -> tuple[int, int]:
    """
    Parses a 'queen-style' move layer (0-55) to determine the destination square.
    These layers encode moves that are similar to how a queen, rook, or bishop moves
    (i.e., straight or diagonal lines for a certain distance).

    Args:
        move_layer (int): The integer representing the specific queen-style move.
                          Ranges from 0 to 55.
        from_rank (int): The 0-indexed rank (row) of the starting square.
        from_file (int): The 0-indexed file (column) of the starting square.

    Returns:
        tuple[int, int]: A tuple containing the 0-indexed rank and file of the
                         destination square (to_rank, to_file).

    Raises:
        ValueError: If the `move_layer` is outside the expected range for
                    queen-style moves.
    """
    to_rank: int = from_rank
    to_file: int = from_file
    distance: int

    # Define the 8 cardinal and intercardinal directions.
    # Each tuple contains (start_layer_index, end_layer_index, rank_change, file_change).
    directions = [
        (0, 6, 1, 0),  # North (dr=1, df=0)
        (7, 13, 1, 1),  # North-East (dr=1, df=1)
        (14, 20, 0, 1),  # East (dr=0, df=1)
        (21, 27, -1, 1),  # South-East (dr=-1, df=1)
        (28, 34, -1, 0),  # South (dr=-1, df=0)
        (35, 41, -1, -1),  # South-West (dr=-1, df=-1)
        (42, 48, 0, -1),  # West (dr=0, df=-1)
        (49, 55, 1, -1),  # North-West (dr=1, df=-1)
    ]

    for start, end, direction_rank, direction_file in directions:
        if QUEEN_MOVE_START <= move_layer <= QUEEN_MOVE_END:  # Ensure move_layer is within the valid range
            if move_layer >= start and move_layer <= end:
                # Calculate the distance from the starting square.
                # Each direction block (e.g., 0-6 for North) represents distances 1 through 7.
                distance = (move_layer - start) + 1
                to_rank = from_rank + distance * direction_rank
                to_file = from_file + distance * direction_file
                return to_rank, to_file

    raise ValueError(f"Invalid move_layer for queen-style move: {move_layer}")


def _parse_knight_move(
    move_layer: int, from_rank: int, from_file: int
) -> tuple[int, int]:
    """
    Parses a knight move layer (56-63) to determine the destination square.
    These layers specifically encode the 8 possible 'L-shaped' moves of a knight.

    Args:
        move_layer (int): The integer representing the specific knight move.
                          Ranges from 56 to 63.
        from_rank (int): The 0-indexed rank (row) of the starting square.
        from_file (int): The 0-indexed file (column) of the starting square.

    Returns:
        tuple[int, int]: A tuple containing the 0-indexed rank and file of the
                         destination square (to_rank, to_file).

    Raises:
        ValueError: If the `move_layer` is outside the expected range for
                    knight moves.
    """
    to_rank: int = from_rank
    to_file: int = from_file

    # Knight moves are hardcoded relative to the starting square.
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
    and the promoted piece type. Underpromotions occur when a pawn promotes to a
    knight, bishop, or rook (not a queen).

    Args:
        move_layer (int): The integer representing the specific underpromotion move.
                          Ranges from 64 to 72.
        from_rank (int): The 0-indexed rank (row) of the starting square of the pawn.
        from_file (int): The 0-indexed file (column) of the starting square of the pawn.

    Returns:
        tuple[int, int, str]: A tuple containing:
            - to_rank (int): The 0-indexed rank of the destination square (always 0 or 7).
            - to_file (int): The 0-indexed file of the destination square.
            - promotion_piece (str): The lowercase single-letter abbreviation of the
                                     promoted piece ('n', 'b', or 'r').

    Raises:
        ValueError: If the `from_rank` is not valid for an underpromotion (i.e.,
                    pawn not on 2nd or 7th rank), or if `move_layer` is invalid.
    """
    to_file: int = from_file
    promotion_piece: str

    # Determine the destination rank (always 8th rank for white, 1st for black)
    if from_rank == 6:  # White pawn on 7th rank (0-indexed rank 6)
        to_rank = 7  # Promotes to 8th rank (0-indexed rank 7)
    elif from_rank == 1:  # Black pawn on 2nd rank (0-indexed rank 1)
        to_rank = 0  # Promotes to 1st rank (0-indexed rank 0)
    else:
        # This indicates an invalid call for an underpromotion move
        raise ValueError(
            f"Invalid from_rank {from_rank} for underpromotion move_layer {move_layer}. "
            "Pawn must be on 2nd or 7th rank."
        )

    # `relative_layer` normalizes the move_layer to be 0-indexed for underpromotions (0-8).
    relative_layer = move_layer - UNDERPROMOTION_MOVE_START

    # Promote to Knight (layers 0-2 relative to UNDERPROMOTION_MOVE_START)
    if relative_layer >= 0 and relative_layer <= 2:
        promotion_piece = "n"  # Knight
        if relative_layer == 0:  # Straight promotion (e.g., e7e8n)
            to_file = from_file
        elif relative_layer == 1:  # Diagonal-left promotion (e.g., e7d8n)
            to_file = from_file - 1
        elif relative_layer == 2:  # Diagonal-right promotion (e.g., e7f8n)
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
    and optional promotion piece. This function serves as the inverse of `encode_move_layer`.

    Args:
        move_layer (int): An integer from 0 to 72 representing the move type,
                          as output by the policy head of the neural network.
        from_rank (int): The 0-indexed rank (row) of the starting square.
        from_file (int): The 0-indexed file (column) of the starting square.

    Returns:
        A tuple (to_rank, to_file, promotion_piece).
        - to_rank (int): The 0-indexed rank of the destination square.
        - to_file (int): The 0-indexed file of the destination square.
        - promotion_piece (str | None): 'n', 'b', 'r' for underpromotions, otherwise None.

    Raises:
        ValueError: If the `move_layer` is outside the expected range (0-72).
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


def _encode_underpromotions(move_distance: int, move: chess.Move) -> int:
    """
    Encodes an underpromotion move into its corresponding integer layer.

    Args:
        move_distance (int): The Manhattan distance between the `from_square` and `to_square`
                             of the pawn promotion (15, 16, or 17).
        move (chess.Move): The chess.Move object representing the underpromotion.

    Returns:
        int: The integer layer (64-72) for the underpromotion.
    """
    # These constants represent the `chess.Square` difference for straight, diagonal-left, and diagonal-right pawn moves.
    # For example, for white, a straight push is +8 (to_square - from_square),
    # a diagonal left capture is +7, and a diagonal right capture is +9.
    # The absolute difference is used to generalize for both white and black pawns.
    STRAIGHT = 8  # Changed from 16 to 8 (abs(chess.SQUARES[file + rank*8 + 8] - chess.SQUARES[file + rank*8]))
    DIAGONAL_LEFT = 7 # Changed from 15 to 7 (abs(chess.SQUARES[file + rank*8 + 7] - chess.SQUARES[file + rank*8]))
    DIAGONAL_RIGHT = 9 # Changed from 17 to 9 (abs(chess.SQUARES[file + rank*8 + 9] - chess.SQUARES[file + rank*8]))

    if move.from_square < move.to_square: # White's turn
        if move.from_square % 8 > move.to_square % 8: # Left capture
            move_distance = 7
        elif move.from_square % 8 < move.to_square % 8: # Right capture
            move_distance = 9
        else: # Straight push
            move_distance = 8
    else: # Black's turn
        if move.from_square % 8 > move.to_square % 8: # Right capture
            move_distance = 9
        elif move.from_square % 8 < move.to_square % 8: # Left capture
            move_distance = 7
        else: # Straight push
            move_distance = 8

    match move.promotion:
        case chess.KNIGHT:
            if move_distance == STRAIGHT:
                return 64
            if move_distance == DIAGONAL_LEFT:
                return 65
            if move_distance == DIAGONAL_RIGHT:
                return 66
        case chess.BISHOP:
            if move_distance == STRAIGHT:
                return 67
            if move_distance == DIAGONAL_LEFT:
                return 68
            if move_distance == DIAGONAL_RIGHT:
                return 69
        case chess.ROOK:
            if move_distance == STRAIGHT:
                return 70
            if move_distance == DIAGONAL_LEFT:
                return 71
            if move_distance == DIAGONAL_RIGHT:
                return 72
    raise ValueError(f"Could not encode underpromotion move: {move}")


def _encode_knight_moves(rank_offset: int, file_offset: int) -> int | None:
    """
    Encodes the relative rank and file offsets of a knight move into its
    corresponding integer layer (56-63).

    Args:
        rank_offset (int): The difference in rank between the destination and starting square.
        file_offset (int): The difference in file between the destination and starting square.

    Returns:
        int | None: The integer layer for the knight move, or None if the offsets
                    do not correspond to a valid knight move.
    """
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
    return None # Should not happen if `encode_move_layer` is called with a valid knight move


def _encode_queen_like_moves(rank_offset: int, file_offset: int) -> int:
    """
    Encodes the relative rank and file offsets of a queen-like move (straight or diagonal)
    into its corresponding integer layer (0-55).

    Args:
        rank_offset (int): The difference in rank between the destination and starting square.
        file_offset (int): The difference in file between the destination and starting square.

    Returns:
        int: The integer layer for the queen-like move.
    """
    QUEEN_DIRECTIONS = [
        (1, 0),  # North
        (1, 1),  # North-East
        (0, 1),  # East
        (-1, 1),  # South-East
        (-1, 0),  # South
        (-1, -1),  # South-West
        (0, -1),  # West
        (1, -1),  # North-West
    ]
    MAX_QUEEN_DISTANCE = 7

    # Determine direction index
    direction_idx = -1
    # Normalize dr, df to get the direction vector (e.g., (1,0) for North)
    # This handles moves of different distances in the same direction.
    norm_dr = 0 if rank_offset == 0 else rank_offset // abs(rank_offset)
    norm_df = 0 if file_offset == 0 else file_offset // abs(file_offset)

    for i, (dir_r, dir_f) in enumerate(QUEEN_DIRECTIONS):
        if norm_dr == dir_r and norm_df == dir_f:
            direction_idx = i
            break
    
    if direction_idx == -1:
        raise ValueError(f"Could not determine direction for offsets: ({rank_offset}, {file_offset})")

    # Determine distance
    # For straight moves (rank_offset=0 or file_offset=0), distance is abs(rank_offset) or abs(file_offset)
    # For diagonal moves, distance is abs(rank_offset) (which is equal to abs(file_offset))
    distance = max(abs(rank_offset), abs(file_offset))
    
    if not (1 <= distance <= MAX_QUEEN_DISTANCE):
        raise ValueError(f"Invalid distance {distance} for queen-like move with offsets: ({rank_offset}, {file_offset})")


    # Queen-style layers are 0-55
    # Layer = (direction_idx * MAX_QUEEN_DISTANCE) + (distance - 1)
    # Example: Direction N (idx 0), distance 1 -> layer 0 (0*7 + 0)
    #          Direction N (idx 0), distance 7 -> layer 6 (0*7 + 6)
    #          Direction NE (idx 1), distance 1 -> layer 7 (1*7 + 0)
    #          Direction NW (idx 7), distance 7 -> layer 7*7 + 6 = 55
    return (direction_idx * MAX_QUEEN_DISTANCE) + (distance - 1)


def encode_move_layer(move: chess.Move) -> int:
    """
    Encodes a legal chess.Move object into its corresponding integer layer
    for the AlphaZero-style policy head (0-72).

    This function categorizes moves into three types: underpromotions,
    knight moves, and "queen-like" moves (straight or diagonal).

    Args:
        move (chess.Move): The legal chess.Move object to encode.

    Returns:
        int: The integer layer (0-72) representing the encoded move.

    Raises:
        ValueError: If the move cannot be encoded (e.g., an invalid move type
                    or an unhandled case).
    """
    # Handle underpromotions (promotion to Knight, Bishop, or Rook)
    # Queen promotions are handled as regular queen-like moves.
    if move.promotion is not None and move.promotion != chess.QUEEN:
        # For underpromotions, the move_distance parameter is implicitly handled
        # within _encode_underpromotions based on the move's to_square and from_square.
        return _encode_underpromotions(0, move) # The 0 for move_distance is a placeholder and not used in the function after modification.

    # Calculate rank and file offsets
    from_rank = chess.square_rank(move.from_square)
    from_file = chess.square_file(move.from_square)

    to_rank = chess.square_rank(move.to_square)
    to_file = chess.square_file(move.to_square)

    rank_offset = to_rank - from_rank
    file_offset = to_file - from_file

    # Check for knight moves
    if (abs(rank_offset) == 1 and abs(file_offset) == 2) or (
        abs(rank_offset) == 2 and abs(file_offset) == 1
    ):
        encoded_knight_move = _encode_knight_moves(rank_offset, file_offset)
        if encoded_knight_move is not None:
            return encoded_knight_move
        else:
            raise ValueError(f"Could not encode knight move: {move}")

    # Check for queen-like moves (straight or diagonal line)
    if rank_offset == 0 or file_offset == 0 or abs(rank_offset) == abs(file_offset):
        return _encode_queen_like_moves(rank_offset, file_offset)

    # If the move is not an underpromotion, knight move, or queen-like move, it's an error.
    # This should ideally not be reached if the input `move` is a legal chess move.
    raise ValueError(f"Could not encode move: {move}. Move type not recognized.")
