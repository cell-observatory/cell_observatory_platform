#!/usr/bin/env python
"""Find unreadable chunks in zarr (v3, sharded) image arrays before a long training run.

The training loader aborts the whole run on the first chunk tensorstore cannot
decode (e.g. "Invalid blosc-compressed data"), so a single corrupt shard on
disk costs one job per epoch. This script opens every ``*.zarr`` array under
one or more roots (or the paths listed in a file) and reads it region by
region on the array's chunk grid, reporting every region whose read raises.

Run it inside the project container on a compute node (I/O bound, no GPU):

    apptainer exec --bind /groups $SIF python scripts/utils/scan_zarr_chunks.py \
        --root /path/to/synthetic_data_iteration_2 --workers 16 --out bad_chunks.txt

    # or a list of tile paths (one per line, absolute or relative to --root)
    apptainer exec --bind /groups $SIF python scripts/utils/scan_zarr_chunks.py \
        --root /path/to/root --tiles tiles.txt --out bad_chunks.txt

Output: one line per unreadable region, ``<zarr path>\t<region start>\t<error>``,
plus a summary on stderr. Feed the bad tiles to
``datasets.databases.exclude_tile_path_patterns``.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import tensorstore as ts


def find_zarr_arrays(root: Path) -> Iterator[Path]:
    """Yield zarr array directories (those holding zarr.json with node_type array)."""
    for dirpath, dirnames, filenames in os.walk(root):
        if "zarr.json" in filenames:
            try:
                import json

                meta = json.loads((Path(dirpath) / "zarr.json").read_text())
            except Exception:  # noqa: BLE001 - unreadable metadata is itself a finding
                yield Path(dirpath)
                dirnames[:] = []
                continue
            if meta.get("node_type") == "array":
                yield Path(dirpath)
                dirnames[:] = []  # arrays hold no nested arrays


def _regions(shape: tuple[int, ...], chunk: tuple[int, ...]) -> Iterator[tuple[slice, ...]]:
    ranges = [range(0, s, c) for s, c in zip(shape, chunk)]
    for starts in itertools.product(*ranges):
        yield tuple(slice(st, min(st + c, s)) for st, c, s in zip(starts, chunk, shape))


def scan_array(path: Path, driver: str = "zarr3", region: str = "shard") -> list[tuple[str, str, str]]:
    """Read every region of one array on the shard grid (default) or the inner chunk grid; return (path, region, error) rows.

    Sharded zarr3 arrays store one file per *write* chunk (the shard, e.g. 13x128x128x128) holding many inner read chunks
    (e.g. 1x32x32x32). Reading on the inner grid costs one shard-index lookup + a small read per inner chunk (~220 k reads
    per tile, hours per array); one read per shard decodes the same bytes in ~264 reads. A corrupt blosc block fails either way.
    """
    bad: list[tuple[str, str, str]] = []
    try:
        arr = ts.open({"driver": driver, "kvstore": {"driver": "file", "path": str(path)}}, read=True).result()
    except Exception as e:  # noqa: BLE001
        return [(str(path), "<open>", repr(e)[:300])]
    shape = tuple(arr.shape)
    if region == "chunk":
        chunk = tuple(arr.chunk_layout.read_chunk.shape or arr.chunk_layout.write_chunk.shape or shape)
    else:
        chunk = tuple(arr.chunk_layout.write_chunk.shape or arr.chunk_layout.read_chunk.shape or shape)
    for region in _regions(shape, chunk):
        try:
            arr[region].read().result()
        except Exception as e:  # noqa: BLE001
            bad.append((str(path), str([s.start for s in region]), repr(e)[:300]))
    return bad


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="root directory to walk (or base for --tiles)")
    ap.add_argument("--tiles", type=Path, default=None, help="file with one zarr path per line instead of walking --root")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("bad_chunks.txt"))
    ap.add_argument("--driver", default="zarr3")
    ap.add_argument("--region", choices=("shard", "chunk"), default="shard", help="read granularity (default: one read per shard)")
    args = ap.parse_args(argv)

    if args.tiles:
        arrays = []
        for line in args.tiles.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line) if line.startswith("/") else args.root / line
            arrays.extend(find_zarr_arrays(p) if (p / "zarr.json").exists() is False else [p])
    else:
        arrays = list(find_zarr_arrays(args.root))
    print(f"[scan] {len(arrays)} arrays under {args.root}", file=sys.stderr)

    t0 = time.time()
    n_bad = 0
    with args.out.open("w") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(scan_array, a, args.driver, args.region): a for a in arrays}
        for i, fut in enumerate(as_completed(futs), 1):
            for row in fut.result():
                n_bad += 1
                out.write("\t".join(row) + "\n")
                out.flush()
            if i % 50 == 0 or i == len(arrays):
                print(f"[scan] {i}/{len(arrays)} arrays, {n_bad} bad regions, {time.time() - t0:.0f}s", file=sys.stderr)
    print(f"[scan] done: {n_bad} unreadable regions -> {args.out}", file=sys.stderr)
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
