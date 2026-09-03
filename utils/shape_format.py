from collections import OrderedDict


def get_spatial_shape(shape: tuple, fmt: str) -> tuple:
    axis_to_value = OrderedDict[str, int](zip(fmt, shape))
    for key, value in axis_to_value.copy().items():
        if key.upper() not in ["Z", "Y", "X"]:
            axis_to_value.pop(key)
    return tuple(axis_to_value.values())
