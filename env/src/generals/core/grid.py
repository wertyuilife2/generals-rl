"""Shared row-major grid geometry for interactive and automated players."""

def adjacent(cell: int, direction: int, height: int, width: int) -> int | None:
    """Return a four-neighbor without allowing row wraparound."""
    if not 0 <= cell < height * width or not 0 <= direction < 4:
        return None
    y, x = divmod(int(cell), width)
    dy, dx = ((-1, 0), (0, 1), (1, 0), (0, -1))[int(direction)]
    y, x = y + dy, x + dx
    return y * width + x if 0 <= y < height and 0 <= x < width else None


def direction_between(source: int, target: int, height: int, width: int) -> int | None:
    for direction in range(4):
        if adjacent(source, direction, height, width) == target:
            return direction
    return None


