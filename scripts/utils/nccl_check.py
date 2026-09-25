"""Intra-node NCCL sanity: all-gather / all-reduce / reduce-scatter bandwidth across all visible GPUs.

    torchrun --nproc_per_node=8 scripts/utils/nccl_check.py [--mb 25] [--iters 20]

Prints per-op median time and effective bus bandwidth. 25 MB ~ one ViT-L block's bf16 params.
"""
import argparse, os, time
import torch, torch.distributed as dist

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mb", type=float, default=25); ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    dist.init_process_group("nccl"); r = dist.get_rank(); w = dist.get_world_size()
    torch.cuda.set_device(r % torch.cuda.device_count()); dev = torch.cuda.current_device()
    n = int(a.mb * 1e6 / 2)  # bf16 elements
    shard = torch.randn(n // w, device=dev, dtype=torch.bfloat16); full = torch.empty(n, device=dev, dtype=torch.bfloat16)
    x = torch.randn(n, device=dev, dtype=torch.float32)
    def bench(name, fn):
        for _ in range(3): fn()
        torch.cuda.synchronize(); dist.barrier(); ts = []
        for _ in range(a.iters):
            torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        ts.sort(); med = ts[len(ts)//2]
        if r == 0: print(f"{name:18s} {a.mb:6.1f} MB  median {med*1e3:8.2f} ms  ({a.mb/1e3/med:7.1f} GB/s payload)", flush=True)
    bench("all_gather bf16", lambda: dist.all_gather_into_tensor(full, shard))
    bench("all_reduce fp32", lambda: dist.all_reduce(x))
    bench("reduce_scatter", lambda: dist.reduce_scatter_tensor(torch.empty(n // w, device=dev), x[: (n // w) * w].view(w, -1).contiguous().view(-1)))
    bench("all_reduce 4B", lambda: dist.all_reduce(torch.ones(1, device=dev)))
    if r == 0: print("env:", {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}, flush=True)
    dist.destroy_process_group()
if __name__ == "__main__": main()
