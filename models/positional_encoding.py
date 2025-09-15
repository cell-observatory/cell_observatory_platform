
import numpy as np

def sincos(embed_dim, pos, temperature=10000, dtype=np.float32):
    """ Returns a matrix of size [sequence_length, embed_dim] """
    exponent = np.arange(embed_dim // 2, dtype=dtype) / (embed_dim / 2.) # (embed_dim//2)
    w = 1. / temperature ** exponent

    pos = pos.reshape(-1)   # (sequence_length, )
    hc = np.einsum('n,c->nc', pos, w)   # (sequence_length, embed_dim//2)
    return np.concatenate([np.sin(hc) , np.cos(hc)], axis=1) # (sequence_length, embed_dim)


def positional_encoding_1d(
    embed_dim,
    sequence_length,
    temperature=10000,
    cls_token=False,
    dtype=None
):
    """
    N = sequence_length for 1D case
    Returns a matrix of size
        if cls_token=True    [N + 1, embed_dim]
        else                 [N, embed_dim]
    """
    dtype = dtype if dtype is not None else np.float32
    pos = np.arange(sequence_length, dtype=dtype)
    emb = sincos(embed_dim=embed_dim, pos=pos, temperature=temperature, dtype=dtype)

    if cls_token:
        return np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    else:
        return emb

def positional_encoding_2d(
    embed_dim,
    lateral_sequence_length,
    temperature=10000,
    cls_token=False,
    dtype=None
):
    """
    N = sequence_length^2
    Returns a matrix of size
        if cls_token=True    [N + 1, embed_dim]
        else                 [N, embed_dim]
    """
    num_dims = 2
    dtype = dtype if dtype is not None else np.float32
    d = int(np.floor(embed_dim / (2*num_dims)) * 2)
    pad = embed_dim - (d * num_dims)

    xgrid = np.arange(lateral_sequence_length, dtype=dtype)
    ygrid = np.arange(lateral_sequence_length, dtype=dtype)
    ygrid, xgrid = np.meshgrid(ygrid, xgrid, indexing='ij')

    yemb = sincos(embed_dim=d, pos=ygrid, temperature=temperature, dtype=dtype)  # (N, d)
    xemb = sincos(embed_dim=d, pos=xgrid, temperature=temperature, dtype=dtype)  # (N, d)
    emb = np.concatenate([yemb, xemb], axis=1)  # (N, d*2)

    if pad > 0:
        emb = np.pad(emb, ((0, 0), (0, pad)), mode='constant', constant_values=0)  # (N, embed_dim)

    if cls_token:
        return np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    else:
        return emb

def positional_encoding_3d(
    embed_dim,
    lateral_sequence_length,
    axial_sequence_length=None,
    temporal_sequence_length=None,
    temperature=10000,
    cls_token=False,
    dtype=None
):
    """
    N = lateral_sequence_length^2 * (axial_sequence_length or temporal_sequence_length)
    Returns a matrix of size
        if cls_token=True    [N + 1, embed_dim]
        else                 [N, embed_dim]
    """
    num_dims = 3
    dtype = dtype if dtype is not None else np.float32
    d = int(np.floor(embed_dim / (2*num_dims)) * 2)
    pad = embed_dim - (d * num_dims)

    if axial_sequence_length is not None and temporal_sequence_length is not None:
        raise ValueError("Use `positional_encoding_4d` if you have both axial and temporal sequence_length")

    xgrid = np.arange(lateral_sequence_length, dtype=dtype)
    ygrid = np.arange(lateral_sequence_length, dtype=dtype)

    if axial_sequence_length is not None:
        zgrid = np.arange(axial_sequence_length, dtype=dtype)
    else:
        zgrid = np.arange(temporal_sequence_length, dtype=dtype)

    zgrid, ygrid, xgrid = np.meshgrid(zgrid, ygrid, xgrid, indexing='ij')

    zemb = sincos(embed_dim=d, pos=zgrid, temperature=temperature, dtype=dtype)  # (N, d)
    yemb = sincos(embed_dim=d, pos=ygrid, temperature=temperature, dtype=dtype)  # (N, d)
    xemb = sincos(embed_dim=d, pos=xgrid, temperature=temperature, dtype=dtype)  # (N, d)
    emb = np.concatenate([zemb, yemb, xemb], axis=1)  # (N, d*3)

    if pad > 0:
        emb = np.pad(emb, ((0, 0), (0, pad)), mode='constant', constant_values=0)  # (N, embed_dim)

    if cls_token:
        return np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    else:
        return emb


def positional_encoding_4d(
    embed_dim,
    lateral_sequence_length,
    axial_sequence_length,
    temporal_sequence_length,
    temperature=10000,
    cls_token=False,
    dtype=None
):
    """
    N = lateral_sequence_length^2 * axial_sequence_length * temporal_sequence_length
    Returns a matrix of size
        if cls_token=True    [N + 1, embed_dim]
        else                 [N, embed_dim]
    """
    num_dims = 4
    dtype = dtype if dtype is not None else np.float32
    d = int(np.floor(embed_dim / (2*num_dims)) * 2)
    pad = embed_dim - (d * num_dims)

    xgrid = np.arange(lateral_sequence_length, dtype=dtype)
    ygrid = np.arange(lateral_sequence_length, dtype=dtype)
    zgrid = np.arange(axial_sequence_length, dtype=dtype)
    tgrid = np.arange(temporal_sequence_length, dtype=dtype)
    tgrid, zgrid, ygrid, xgrid = np.meshgrid(tgrid, zgrid, ygrid, xgrid, indexing='ij')

    temb = sincos(embed_dim=d, pos=tgrid, temperature=temperature, dtype=dtype)  # (N, d)
    zemb = sincos(embed_dim=d, pos=zgrid, temperature=temperature, dtype=dtype)  # (N, d)
    yemb = sincos(embed_dim=d, pos=ygrid, temperature=temperature, dtype=dtype)  # (N, d)
    xemb = sincos(embed_dim=d, pos=xgrid, temperature=temperature, dtype=dtype)  # (N, d)
    emb = np.concatenate([temb, zemb, yemb, xemb], axis=1)  # (N, d*4)

    if pad > 0:
        emb = np.pad(emb, ((0, 0), (0, pad)), mode='constant', constant_values=0)  # (N, embed_dim)

    if cls_token:
        return np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    else:
        return emb


if __name__ == '__main__':
    embed_dim = 64
    d = int(np.floor(embed_dim / 8) * 2)

    x = np.linspace(0, 40, 4)
    y = np.linspace(0, 50, 5)
    z = np.linspace(0, 60, 6)
    t = np.linspace(0, 70, 7)

    print(f"1D: {x.shape}")

    yy, xx = np.meshgrid(y, x, indexing='ij')
    print(f"2D: {yy.shape}")

    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
    print(f"3D: {zz.shape}")

    tt, zz, yy, xx = np.meshgrid(t, z, y, x, indexing='ij')
    print(f"4D: {tt.shape}")

    temb = sincos(embed_dim=d, pos=tt)
    zemb = sincos(embed_dim=d, pos=zz)
    yemb = sincos(embed_dim=d, pos=yy)
    xemb = sincos(embed_dim=d, pos=xx)
    print(f"{temb.shape=}, {zemb.shape=}, {yemb.shape=}, {xemb.shape=}")

    emb = np.concatenate([temb, zemb, yemb, xemb], axis=1)
    print(f"sequence_length: {np.prod([t.shape, z.shape, y.shape, x.shape])}")
    print(f"{emb.shape=}")
