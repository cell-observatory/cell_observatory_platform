from enum import Enum

class Dimension(Enum):
    DIM_3D_BZYX   = "BZYX"
    DIM_4D_BZYXC  = "BZYXC"
    DIM_4D_BTZYX  = "BTZYX"
    DIM_5D_BTZYXC = "BTZYXC"


class ColorMode(Enum):
    AVG = "Average"
    MATCH = "Matching: skip if color channel not matching"
    # TODO: add different color modes: index, target protein etc

class DataConfig:
    def __init__(self, t = None, z = 128, y = 128, x = 128, c = None):
        self.t = t
        self.z = z
        self.y = y
        self.x = x
        self.c = c

        if c is None:
            self.color_mode = ColorMode.AVG
        else:
            self.color_mode = ColorMode.MATCH

        self.dim = self._determine_data_dimension()

    def _determine_data_dimension(self):
        has_time = self.t is not None
        has_color = self.c is not None

        if has_time and has_color:
            return Dimension.DIM_5D_BTZYXC
        elif has_time and not has_color:
            return Dimension.DIM_4D_BTZYX
        elif has_color and not has_time:
            return Dimension.DIM_4D_BZYXC
        else:
            return Dimension.DIM_3D_BZYX