import math

TRITON_MAX_NUMEL = 2**20
BLOCK_SIZE = 256


def is_power_of_two(x):
    return x > 0 and not x & (x - 1)


def next_power_of_two(x):
    """Return the smallest power of two greater than x."""
    x = int(math.ceil(x))
    return 1 << (x - 1).bit_length()
