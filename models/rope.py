
# based on: https://github.com/naver-ai/rope-vit and extended to 3D and 4D
# NOTE: the divisibility assertions may be too strong, we might have to relax and pad or similar

import sys
import logging
from typing import Optional, Tuple

import torch

logging.basicConfig(
	stream=sys.stdout,
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



def generate_frequency_spectrum(dim: int, 
                                num_heads: int, 
                                theta: float = 10.0, 
                                random_rotation_per_head: bool = True,
                                input_fmt: str = "TZYXC",
                                device: str = 'cuda',
                                dtype: torch.dtype = torch.bfloat16
):
    if input_fmt == "YXC":
        # assert dim % 4 == 0, "head_dim must be divisible by 4 for 2D ROPE."
        freqs_x, freqs_y = [], []
        # generate frequency spectrum: 1 / (theta ** (4i / d)) for i = 0, ..., d/4 - 1
        mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
        for i in range(num_heads):
            # we can either have a random rotation per head, or the same rotation for all heads
            # if random rotation per head: sample uniform distribution on the interval [0,2pi)
            # this way each attention head gets its own 2-D orientation
            # so different heads are initialized with different basis sets
            angles = torch.rand(1) * 2 * torch.pi if random_rotation_per_head else torch.zeros(1)
            # form: (cosϕ,-sinϕ) and (sinϕ,cosϕ) where we use (cos(ϕ+π/2​),sin(ϕ+π/2​))=(−sinϕ,cosϕ)
            # if zeros we get (1,0) and (0,1) as in the standard RoPE
            # after referring to compute_mixed_cis, we see that this implies that the sequence is 
            # given by [mag_k(x_i*cosϕ + y_i*sinϕ),mag_k(-x_i*sinϕ+y_i*cosϕ)]
            fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
            fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
            freqs_x.append(fx)
            freqs_y.append(fy)
        freqs_x = torch.stack(freqs_x, dim=0)
        freqs_y = torch.stack(freqs_y, dim=0)
        # freqs: (2, num_heads, dim//4 * 2)
        freqs = torch.stack([freqs_x, freqs_y], dim=0)
    
    elif input_fmt == "TYXC" or input_fmt == "ZYXC":
        # the below follows the logic as above but generalized to 3D
        # assert dim % 6 == 0, "head_dim must be divisible by 6 for 3D ROPE."
        J = dim // 6

        base = torch.arange(0, dim, 6)[: (dim // 6)].float() / dim
        mag = theta ** (-base)

        # freqs: (3, num_heads, dim//6*3)
        freqs = torch.empty(3, num_heads, J*3)

        for h in range(num_heads):
            if random_rotation_per_head:
                M3 = torch.randn(3, 3)
                # generate 3 orthonormal basis vectors 
                # in R^3 from qr decomposition A = QR
                Q, _ = torch.linalg.qr(M3)
            else:
                Q = torch.eye(3)

            # build 3 blocks of length J
            blocks = []
            for k in range(3):
                # broadcast: [3,1] * [1,J] -> [3,J]
                blocks.append(Q[:, [k]] @ mag[None, :])
            # blocks: [3, 3, J] -> freqs: [3, num_heads, dim//6*3]
            freqs[:, h, :] = torch.cat(blocks, dim=-1)
    
    elif input_fmt == "TZYXC":
        # the below follows the logic as above but generalized to 4D
        # assert dim % 8 == 0, "head_dim must be divisible by 8 for 4D ROPE."
        J = dim // 8

        base = torch.arange(0, dim, 8)[: (dim // 8)].float() / dim
        mag = theta ** (-base)

        # freqs: (4, num_heads, dim//8*4)
        freqs = torch.empty(4, num_heads, J*4)

        for h in range(num_heads):
            if random_rotation_per_head:
                M4 = torch.randn(4, 4)
                # generate 4 orthonormal basis vectors 
                # in R^4 from qr decomposition A = QR
                Q, _ = torch.linalg.qr(M4)
            else:
                Q = torch.eye(4)

            # build 4 blocks of length J
            blocks = []
            for k in range(4):
                # broadcast: [4,1] * [1,J] -> [4,J]
                blocks.append(Q[:, [k]] @ mag[None, :])
            # freqs: [4, num_heads, dim//8*4]
            freqs[:, h, :] = torch.cat(blocks, dim=-1)

    else:
        raise NotImplementedError(f"Unknown input_fmt={input_fmt}")
    
    return freqs.to(dtype=dtype, device=device)

def generate_grid_indices(
    end_x: int,
    end_y: int,
    end_z: Optional[int] = None,
    end_t: Optional[int] = None,
    input_fmt: str = "TZYXC",
    device: str = 'cuda',
    dtype: torch.dtype = torch.bfloat16
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    need_T = "T" in input_fmt
    need_Z = "Z" in input_fmt
    need_Y = "Y" in input_fmt
    need_X = "X" in input_fmt

    assert need_X and need_Y, "X and Y must be present in all supported formats."

    T = int(end_t) if need_T else 1
    Z = int(end_z) if need_Z else 1
    Y = int(end_y)
    X = int(end_x)

    N = T * Z * Y * X

    idx = torch.arange(N)
    x = (idx % X)
    y = ((idx // X) % Y)
    z = None
    t = None

    if need_Z:
        z = ((idx // (X * Y)) % Z)
    if need_T:
        t = (idx // (X * Y * (Z if need_Z else 1)))

    t = t.to(dtype=dtype, device=device) if t is not None else None
    z = z.to(dtype=dtype, device=device) if z is not None else None
    y = y.to(dtype=dtype, device=device)
    x = x.to(dtype=dtype, device=device)

    return (t, z, y, x)

def compute_mixed_cis(freqs: torch.Tensor,
                      num_heads: int,
                      t_x: torch.Tensor,
                      t_y: torch.Tensor,
                      t_z: Optional[torch.Tensor] = None,
                      t_t: Optional[torch.Tensor] = None,
                      input_fmt: str = "TZYXC"):
    
    def _outer_pos_freq(pos, f):     
        if pos.dim() == 1:
            # [N,1] * [H,1,J] -> [H,N,J]
            return (pos.unsqueeze(-1) @ f.unsqueeze(-2))
        else:
            # broadcast: [B,1,N,1] * [1,H,1,J] -> [B,H,N,J]
            return (pos[:, None, :, None] * f[None, :, None, :])

    with torch.amp.autocast(enabled=False, device_type='cuda'):
        if input_fmt == "YXC":
            # [H,N,Jx] or [B,H,N,Jx]
            fx = _outer_pos_freq(t_x, freqs[0])
            # [H,N,Jy] or [B,H,N,Jy]
            fy = _outer_pos_freq(t_y, freqs[1])
            phases = fx + fy
        elif input_fmt == "TYXC":
            assert t_t is not None, "t_t must be provided for TYXC format"
            fx = _outer_pos_freq(t_x, freqs[0])
            fy = _outer_pos_freq(t_y, freqs[1])
            ft = _outer_pos_freq(t_t, freqs[2])
            phases = fx + fy + ft
        elif input_fmt == "ZYXC":
            assert t_z is not None, "t_z must be provided for ZYXC format"
            fx = _outer_pos_freq(t_x, freqs[0])
            fy = _outer_pos_freq(t_y, freqs[1])
            fz = _outer_pos_freq(t_z, freqs[2])
            phases = fx + fy + fz
        elif input_fmt == "TZYXC":
            assert t_t is not None and t_z is not None, "t_t and t_z must be provided for TZYXC format"
            fx = _outer_pos_freq(t_x, freqs[0])
            fy = _outer_pos_freq(t_y, freqs[1])
            fz = _outer_pos_freq(t_z, freqs[2])
            ft = _outer_pos_freq(t_t, freqs[3])
            phases = fx + fy + fz + ft
        else:
            raise NotImplementedError

        if phases.dtype != torch.float32: # polar doesn't support bf16
            dtype = phases.dtype
            ones = torch.ones_like(phases, dtype=torch.float32)
            freqs_cis = torch.polar(ones, phases.to(torch.float32)).to(dtype)
        else:
            ones = torch.ones_like(phases)
            freqs_cis = torch.polar(ones, phases)
            
        if freqs_cis.dim() == 3:
            return freqs_cis
        else:
            return freqs_cis

def compute_axial_cis(dim: int, 
                      end_x: int, 
                      end_y: int,
                      end_z: int, 
                      end_t: int,
                      input_fmt: str = "TZYXC", 
                      theta: float = 100.0,
                      device: str = 'cuda'
):
    # NOTE: in the paper they define: R(n,2t)=e{iθ_{t}​p^{n}_{x}​,R(n,2t+1)=eθ_{t}​p^{n}_{y}
    #       however in the reference code the assignment per embedding dimension is:
    #       [x-slot, x-slot, ..., y-slot, y-slot, ...] i.e. the specific assignment of 
    #       x,y positions to dimensions is not interleaved but blockwise

    if input_fmt == "YXC":
        # assert dim % 4 == 0, "head_dim must be divisible by 4 for 2D ROPE."
        mag = 1.0 / (theta ** (torch.arange(0, dim, 4, device=device)[: (dim // 4)].float() / dim))
        
        t_t, t_z, t_y, t_x = generate_grid_indices(end_x=end_x, 
                                                   end_y=end_y, 
                                                   input_fmt=input_fmt, 
                                                   device=device,
                                                   dtype=torch.float32)

        freqs_x = torch.outer(t_x, mag)
        freqs_y = torch.outer(t_y, mag)

    elif input_fmt == "TYXC":
        # assert dim % 6 == 0, "head_dim must be divisible by 6 for 3D ROPE."
        base = torch.arange(0, dim, 6, device=device)[: (dim // 6)].float() / dim
        mag = theta ** (-base)
        
        t_t, t_z, t_y, t_x = generate_grid_indices(end_x=end_x, 
                                                   end_y=end_y, 
                                                   end_t=end_t, 
                                                   input_fmt=input_fmt,
                                                   device=device,
                                                   dtype=torch.float32)
        
        freqs_x = torch.outer(t_x, mag)
        freqs_y = torch.outer(t_y, mag)
        freqs_t = torch.outer(t_t, mag)

    elif input_fmt == "ZYXC":
        # assert dim % 6 == 0, "head_dim must be divisible by 6 for 3D ROPE."
        base = torch.arange(0, dim, 6, device=device)[: (dim // 6)].float() / dim
        mag = theta ** (-base)
        
        t_t, t_z, t_y, t_x = generate_grid_indices(end_x=end_x, 
                                                   end_y=end_y, 
                                                   end_z=end_z, 
                                                   input_fmt=input_fmt,
                                                   device=device,
                                                   dtype=torch.float32)
        
        freqs_x = torch.outer(t_x, mag)
        freqs_y = torch.outer(t_y, mag)
        freqs_z = torch.outer(t_z, mag)

    elif input_fmt == "TZYXC":
        # assert dim % 8 == 0, "head_dim must be divisible by 8 for 4D ROPE."
        base = torch.arange(0, dim, 8, device=device)[: (dim // 8)].float() / dim
        mag = theta ** (-base)
        
        t_t, t_z, t_y, t_x = generate_grid_indices(end_x=end_x, 
                                                   end_y=end_y, 
                                                   end_z=end_z, 
                                                   end_t=end_t, 
                                                   input_fmt=input_fmt,
                                                   device=device,
                                                   dtype=torch.float32)
        
        freqs_x = torch.outer(t_x, mag)
        freqs_y = torch.outer(t_y, mag)
        freqs_z = torch.outer(t_z, mag)
        freqs_t = torch.outer(t_t, mag)

    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)

    if input_fmt == "YXC":
        return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

    if input_fmt == "TYXC":
        freqs_cis_t = torch.polar(torch.ones_like(freqs_t), freqs_t)
        return torch.cat([freqs_cis_x, freqs_cis_y, freqs_cis_t], dim=-1)
    
    elif input_fmt == "ZYXC":
        freqs_cis_z = torch.polar(torch.ones_like(freqs_z), freqs_z)
        return torch.cat([freqs_cis_x, freqs_cis_y, freqs_cis_z], dim=-1)

    elif input_fmt == "TZYXC":
        freqs_cis_z = torch.polar(torch.ones_like(freqs_z), freqs_z)
        freqs_cis_t = torch.polar(torch.ones_like(freqs_t), freqs_t)
        return torch.cat([freqs_cis_x, freqs_cis_y, freqs_cis_z, freqs_cis_t], dim=-1)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # freqs_cis: (N, J) branch
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        # freq_cis reshaped to (1, 1, N, J) since x: [B, H, N, J]
        shape = [d if i >= x.ndim-2 else 1 for i, d in enumerate(x.shape)]
    # freqs_cis: (H, N, J) branch
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        # freq_cis reshaped to (1, H, N, J) since x: [B, H, N, J]
        shape = [d if i >= x.ndim-3 else 1 for i, d in enumerate(x.shape)]
    # freqs_cis: (B, N, J) branch
    elif freqs_cis.shape == (x.shape[0], x.shape[-2], x.shape[-1]):
        shape = [x.shape[0], 1, x.shape[-2], x.shape[-1]]
    # freqs_cis: (B, H, N, J) branch
    elif freqs_cis.shape == (x.shape[0], x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [x.shape[0], x.shape[-3], x.shape[-2], x.shape[-1]]
    else:
        raise ValueError(f"Unexpected freqs_cis shape: {freqs_cis.shape} for x shape: {x.shape}")
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    # xq: [B,H,N,D]
    Jf = freqs_cis.shape[-1]
    De = Jf * 2

    # split off the tail that cannot be rotated cleanly
    xq_even = xq[..., :De]
    xk_even = xk[..., :De]
    xq_tail = xq[..., De:]
    xk_tail = xk[..., De:]
    
    # xq[:-1]: [B, H, N] => xq reshape: [B, H, N, J, 2] for J=D/2 
    # thus xq_: [B, H, N, J] complex and similar for xk
    xq_ = torch.view_as_complex(xq_even.float().reshape(*xq_even.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk_even.float().reshape(*xk_even.shape[:-1], -1, 2))
    
    # if [N, J] -> reshaped to [1, 1, N, J]
    # if [H, N, J] -> reshaped to [1, H, N, J]
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_).to(xq_.device)
    
    # xq_ * freqs_cis: elementwise complex mult -> [B, H, N, J]
    # then view_as_real -> [B, H, N, J, 2] -> flatten last two dims -> [B, H, N, J]
    xq_rot = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_rot = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    if xq_tail.numel():
        xq_out = torch.cat([xq_rot, xq_tail], dim=-1)
        xk_out = torch.cat([xk_rot, xk_tail], dim=-1)
    else:
        xq_out = xq_rot
        xk_out = xk_rot

    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)

