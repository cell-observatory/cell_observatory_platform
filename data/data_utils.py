from itertools import product

from .data_config import DataConfig, ColorMode

def index_mapper(shape: tuple[int, int, int, int, int ,int],
                 batch_config : DataConfig) -> list[tuple[int, int, int, int, int, int]]:
    """
    Given a tensorstore object shape and data config which contains batch shape information, returns a list of tuples
    that map an index to
    Args:
        shape: Tensorstore object shape
        batch_config: DataConfig object which contains batch shape information and how to handle color channels

    Returns:
        indices: list of indices that map a batch index to the tile index and {time,z,y,x,c} slices
    """
    # Tensorstore object dimensions assumed to be in (N,T,Z,Y,X,C) format
    n_tile, n_time, n_z, n_y, n_x, n_c = shape

    # Calculate the number of batches in each store
    if batch_config.color_mode == ColorMode.MATCH and n_c != batch_config.c:
        return None

    if batch_config.color_mode == ColorMode.AVG or  batch_config.color_mode == ColorMode.MATCH:
        # AVG: output will be averaged so will have a single color channel
        # MATCH: output channel size must match input channel size, therefore color channel won't be s
        n_c = 1
    else:
        raise NotImplementedError(f"color mode {batch_config.color_mode} is not supported")

    n_time = n_time // batch_config.t
    n_z = n_z // batch_config.z
    n_y = n_y // batch_config.y
    n_x = n_x // batch_config.x

    indices = list(product(range(n_tile), range(n_time), range(n_z), range(n_y), range(n_x), range(n_c)))

    return indices

def middle_out_crop_start_index(shape: tuple[int, int, int, int, int ,int], batch_config : DataConfig) -> tuple[int, int]:
    """
    Due to increased optical performance on axis, data should be cropped from middle out
    Args:
        shape: Tensorstore object shape
        batch_config: DataConfig object which contains batch shape information and how to handle color channels

    Returns:
        (x0, y0): Pixel offset to achieve middle out crop
    """
    # Tensorstore object dimensions assumed to be in (N,T,Z,Y,X,C) format
    n_tile, n_time, n_z, n_y, n_x, n_c = shape

    y0 = (n_y % batch_config.y) // 2
    x0 = (n_x % batch_config.x) // 2

    return y0, x0
