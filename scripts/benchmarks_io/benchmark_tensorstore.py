import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_parent_dir = Path(__file__).resolve().parent.parent.parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

import logging

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorstore as ts
from omegaconf import DictConfig

from cell_observatory_platform.data.data_types import NUMPY_DTYPES
from cell_observatory_platform.data.io import load_hypercubes_dataframe, read_zarr
from cell_observatory_platform.utils import cli
from cell_observatory_platform.utils.common import multiprocess
from cell_observatory_platform.utils.profiling import enable_profiling, pprof_func

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoadingBenchmark:
    def __init__(
        self,
        outdir: Path,
        hypercubes_dataframe_path: str,
        server_folder_path: Optional[str] = None,
        max_rois: int = 100,
        max_tiles: int = 100,
        max_hypercubes: int = 100,
        hpf_list: Optional[List[str]] = None,
        roi_list: Optional[List[str]] = None,
        tile_list: Optional[List[str]] = None,
        occupancy_threshold: float = 0.0,
        dtype: str = "fp16",
    ):
        self.dtype = dtype
        self.bytes_per_sample = np.dtype(NUMPY_DTYPES[self.dtype].value).itemsize

        hypercubes_dataframe_path = Path(hypercubes_dataframe_path)
        if not hypercubes_dataframe_path.exists():
            raise FileNotFoundError(hypercubes_dataframe_path)

        self.server_folder_path = str(server_folder_path) if server_folder_path else None
        self.hypercubes_dataframe, self.hypercubes_dataframe_config = load_hypercubes_dataframe(
            hypercubes_dataframe_path=hypercubes_dataframe_path,
            server_folder_path=server_folder_path,
            max_rois=max_rois,
            max_tiles=max_tiles,
            max_hypercubes=None,
            hpf_list=hpf_list,
            roi_list=roi_list,
            tile_list=tile_list,
            occupancy_threshold=occupancy_threshold
        )

        self.channel_size = self.hypercubes_dataframe.channel_size.values[0]
        self.cube_size = self.hypercubes_dataframe.cube_size.values[0]
        self.time_size = self.hypercubes_dataframe.time_size.values[0]
        self.hypercube_shape = (self.time_size, self.cube_size, self.cube_size, self.cube_size, self.channel_size)

        self.gb_per_hypercube = np.prod(self.hypercube_shape) * self.bytes_per_sample / (1024 ** 3)
        self.output_path = outdir / f"{'_'.join(map(str, self.hypercube_shape))}_{self.dtype}"

        os.makedirs(self.output_path.parent, exist_ok=True)
        
        # self.paths = {
        #     os.path.join(sf, of, tn)
        #     for sf, of, tn in 
        #     zip(self.hypercubes_dataframe["server_folder"], 
        #         self.hypercubes_dataframe["output_folder"], 
        #         self.hypercubes_dataframe["tile_name"]
        #     )
        # }
        
        # self._zarr_handles_data = {
        #     p: read_zarr(p, dtype=self.dtype)
        #     for p in self.paths
        # }
        
        self.results = []

    def slice_hypercube(self, data_tensor, meta: Dict[str, Any]):
        t = slice(meta["time_start"], meta["time_start"] + meta["time_size"])
        c = slice(0, meta["channel_size"])
        z = slice(meta["z_start"], meta["z_start"] + meta["cube_size"])
        y = slice(meta["y_start"], meta["y_start"] + meta["cube_size"])
        x = slice(meta["x_start"], meta["x_start"] + meta["cube_size"])
        return data_tensor[t, z, y, x, c].read()
    
    def _get_ts_context(file_io_limit=128, copy_limit=128) -> ts.Context:
        return ts.Context({
            # optional stuff if not use defaults
            # "file_io_concurrency":   {"limit": file_io_limit},
            # "data_copy_concurrency": {"limit": copy_limit},
            "cache_pool": {"total_bytes_limit": 0}
        })
    
    @pprof_func(label="read_hypercube_ts_benchmark")
    def read_hypercube_from_zarr(self, rec) -> float:
        context = self._get_ts_context()
        
        start = time.perf_counter()
        
        results = []
        for f in rec:
            # zarr_handle = self._zarr_handles_data[os.path.join(f["server_folder"], f["output_folder"], f["tile_name"])]
            zarr_handle = read_zarr(
                image_path=os.path.join(f["server_folder"], f["output_folder"], f["tile_name"]),
                dtype=self.dtype,
                context=context
            )
            results.append(self.slice_hypercube(zarr_handle, f))
        
        results = [r.result() for r in results]
        read_time = time.perf_counter() - start
        nbytes = results[0].nbytes * len(results)
        return read_time, nbytes
    
    def benchmark_parallel_reads(
        self,
        func: callable,
        hypercube_list: List[tuple],
        num_workers: int,
    ) -> Dict[str, float]:

        start_time = time.perf_counter()
        
        batches = np.array_split(hypercube_list, num_workers)
        logger.info(f"Benchmarking with {num_workers} CPU cores with a batch of {len(batches[0])} per worker...")
        
        results = multiprocess(
            func=func,
            jobs=batches,
            cores=num_workers,
            desc=f"Loading hypercubes using {num_workers=} cpu(s)",
            unit='hypercube',
            unit_scale=len(batches[0])
        )

        total_time = time.perf_counter() - start_time

        individual_times = [r[0] for r in results]
        total_bytes = sum(r[1] for r in results)
        total_gb = total_bytes / (1024 ** 3)
        read_time = sum(individual_times)

        return {
            'total_time': total_time,
            'read_time': read_time,
            'total_gb': total_gb,
            'throughput_gb_per_sec': total_gb / total_time,
            'read_throughput_gb_per_sec': total_gb / read_time,
            'throughput_gb_per_sec_per_core': (total_gb / total_time) / num_workers,
            'read_throughput_gb_per_sec_per_core': (total_gb / read_time) / num_workers,
            'hypercubes_per_sec': len(hypercube_list) / total_time,
            'min_time': np.min(individual_times),
            'max_time': np.max(individual_times),
            'mean_time': np.mean(individual_times),
            'std_time': np.std(individual_times),
        }

    def plot_scaling(self):
        if not self.results:
            logger.warning("No results to plot")
            return

        df = pd.DataFrame(self.results)

        # plt.style.use("default")
        plt.style.use("dark_background")
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
        })

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        cpu_cores = df['cpu_count'].values
        total_throughput = df['throughput_gb_per_sec'].values
        per_core_throughput = df['throughput_gb_per_sec_per_core'].values

        ax1.plot(
            cpu_cores, total_throughput, 'o-', color='C0', linewidth=2.5, markersize=8,
            label=f'Total {self.hypercube_shape}, {self.dtype}'
        )

        # ideal_scaling = per_core_throughput[0] * cpu_cores
        # ax1.plot(cpu_cores, ideal_scaling, '--', color='gray', alpha=0.7,linewidth=2, label='Ideal linear scaling')
        ax1.set_xlabel('CPUs')
        ax1.set_ylabel('Throughput (GB/s)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='lower right', frameon=False)
        ax1.set_xticks(cpu_cores)

        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)

        ax1_twin = ax1.twinx()
        convert_to_hypercubes = lambda s: s / self.gb_per_hypercube
        ymin, ymax = ax1.get_ylim()
        ax1_twin.set_ylim((convert_to_hypercubes(ymin), convert_to_hypercubes(ymax)))
        ax1_twin.plot([], [])
        ax1_twin.set_ylabel('Hypercubes/s', color='C0')
        ax1_twin.tick_params(axis='y', labelcolor='C0')
        ax1_twin.spines['right'].set_visible(True)
        ax1_twin.spines['top'].set_visible(False)
        ax1_twin.set_xticks(cpu_cores)

        efficiency_percent = (per_core_throughput / per_core_throughput[0]) * 100

        ax2.plot(
            cpu_cores, per_core_throughput,
            's-', color='C1', linewidth=2.5, markersize=8, label='Total GB/s'
        )

        ax2_twin = ax2.twinx()
        ax2_twin.plot(
            cpu_cores, efficiency_percent,
            '^-', color='C2', linewidth=2.5, markersize=8, label='Scaling efficiency (%)'
        )

        ax2.set_xlabel('CPUs')
        ax2.set_ylabel('Throughput per core (GB/s)', color='C1')
        ax2_twin.set_ylabel('Efficiency (%)', color='C2')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(cpu_cores)

        ax2.legend(loc='upper left', frameon=False)
        ax2_twin.legend(loc='upper right', frameon=False)

        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2_twin.spines['top'].set_visible(False)

        plt.tight_layout()

        for fmt in ['png', 'pdf']:
            dpi = 300 if fmt == 'png' else None
            plt.savefig(self.output_path.with_suffix(f".{fmt}"), bbox_inches='tight', pad_inches=0.25, dpi=dpi)

        plt.close()

    def benchmark_cpu_scaling(self, cpu_counts: List[int],  num_hypercubes: int = 20) -> None:
        logger.info(f"Starting benchmark...")

        for cpu_count in cpu_counts:
            hypercubes_list = self.hypercubes_dataframe.sample(n=num_hypercubes).to_dict(orient='records')

            stats = self.benchmark_parallel_reads(
                hypercube_list=hypercubes_list,
                num_workers=cpu_count,
                func=self.read_hypercube_from_zarr
            )

            self.results.append({
                'cpu_count': cpu_count,
                'num_hypercubes': num_hypercubes,
                'total_time': stats['total_time'],
                'read_time': stats['read_time'],
                'total_gb': stats['total_gb'],
                'throughput_gb_per_sec': stats['throughput_gb_per_sec'],
                'read_throughput_gb_per_sec': stats['read_throughput_gb_per_sec'],
                'throughput_gb_per_sec_per_core': stats['throughput_gb_per_sec_per_core'],
                'read_throughput_gb_per_sec_per_core': stats['read_throughput_gb_per_sec_per_core'],
                'hypercubes_per_sec': stats['hypercubes_per_sec'],
                'min_read_time': stats['min_time'],
                'max_read_time': stats['max_time'],
                'mean_read_time': stats['mean_time'],
                'std_read_time': stats['std_time'],
            })

            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(
                f"CPU {cpu_count}: "
                f"{stats['throughput_gb_per_sec']:.2f} GB/s total, "
                f"{stats['throughput_gb_per_sec_per_core']:.3f} GB/s per core, "
                f"{stats['hypercubes_per_sec']:.2f} hypercubes/s, "
                f"({self.gb_per_hypercube:.3f} GB per hypercube)"
            )

            df = pd.DataFrame(self.results)
            df.to_csv(self.output_path.with_suffix(".csv"), index=False)

            logger.info(f"{'Cores':<10} {'Total GB/s':<12} {'GB/s per Core':<15} {'Efficiency %':<12} {'Hypercubes/s':<10}")
            logger.info("-" * 90)

            baseline_per_core = self.results[0]['throughput_gb_per_sec_per_core'] if self.results else 0

            for result in self.results:
                efficiency = (result['throughput_gb_per_sec_per_core'] / baseline_per_core * 100) if baseline_per_core > 0 else 0
                logger.info(
                    f"{result['cpu_count']:<10} "
                    f"{result['throughput_gb_per_sec']:<12.2f} "
                    f"{result['throughput_gb_per_sec_per_core']:<15.3f} "
                    f"{efficiency:<12.1f} "
                    f"{result['hypercubes_per_sec']:<10.1f}"
                )

            self.plot_scaling()

def benchmark_tensorstore(cfg: DictConfig):
    enable_profiling(cfg)
    benchmarker = DataLoadingBenchmark(
        hypercubes_dataframe_path=cfg.datasets.hypercubes_dataframe_path,
        max_rois=cfg.datasets.max_rois,
        max_tiles=cfg.datasets.max_tiles,
        max_hypercubes=cfg.datasets.max_hypercubes,
        hpf_list=cfg.datasets.hpf_list,
        roi_list=cfg.datasets.roi_list,
        tile_list=cfg.datasets.tile_list,
        server_folder_path=cfg.paths.server_folder_path,
        occupancy_threshold=cfg.datasets.occupancy_threshold,
        outdir=Path(cfg.paths.outdir),
        dtype=cfg.dataset_dtype,
    )
    benchmarker.benchmark_cpu_scaling(
        cpu_counts=cfg.cpu_counts,
        num_hypercubes=cfg.random_hypercubes
    )


def main():
    parser = cli.argparser()
    parser.add_argument("ifile", help="Path to either zarr file or hypercubes dataframe csv")
    parser.add_argument("--hypercube-shape", nargs=5, type=int,
                       help="Hypercube shape as 5 integers: T Z Y X C (required when using --zarr-file)")
    parser.add_argument("--cpu-counts", nargs="+", type=int, default=list(range(1, 17)))
    parser.add_argument("--num-hypercubes", type=int, default=100)
    parser.add_argument("--random-hypercubes", type=int, default=20)
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--server-folder-path", type=str, default='/groups/betzig/betziglab/CellObservatoryData')

    args = parser.parse_args()

    ifile = Path(args.ifile)
    if not ifile.exists():
        logger.error(f"Path does not exist: {ifile}")
        sys.exit(1)

    outdir = Path(__file__).parent.parent / 'benchmarks'
    outdir.mkdir(parents=True, exist_ok=True)

    benchmarker = DataLoadingBenchmark(
        hypercubes_dataframe_path=ifile,
        max_rois=None,
        max_tiles=None,
        max_hypercubes=args.num_hypercubes,
        hpf_list=None,
        roi_list=None,
        tile_list=None,
        server_folder_path=args.server_folder_path,
        occupancy_threshold=0,
        outdir=outdir,
        dtype=args.dtype,
    )
    benchmarker.benchmark_cpu_scaling(
        cpu_counts=args.cpu_counts,
        num_hypercubes=args.random_hypercubes
    )

    print(f"\nBenchmark complete! Results saved to {outdir}")


if __name__ == "__main__":
    main()