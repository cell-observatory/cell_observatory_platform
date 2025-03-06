from itertools import product

from .data_config import DataConfig, ColorMode

def index_mapper(shape: tuple[int, int, int, int, int ,int],
                 data_config : DataConfig) -> list[tuple[int, int, int, int, int, int]]:
    """
    Given a tensorstore object shape and data config which contains batch shape information, returns a list of tuples
    that map an index to
    Args:
        shape: Tensorstore object shape
        data_config: DataConfig object which contains batch shape information and how to handle color channels

    Returns:

    """
    # Tensorstore object dimensions assumed to be in (N,T,Z,Y,X,C) format
    n_tile, n_time, n_z, n_y, n_x, n_c = shape

    # Calculate the number of batches in each store
    color_channel_must_match = data_config.color_mode == ColorMode.MATCH
    if color_channel_must_match and n_c != data_config.c:
        return None

    if data_config.color_mode == ColorMode.AVG or color_channel_must_match:
        n_c = 1
    else:
        raise NotImplementedError(f"color mode {data_config.color_mode} is not supported")

    has_time = data_config.t is not None
    if has_time:
        n_time = n_time // data_config.t

    n_z = n_z // data_config.z
    n_y = n_y // data_config.y
    n_x = n_x // data_config.x

    return list(product(range(n_tile), range(n_time), range(n_z), range(n_y), range(n_x), range(n_c)))
