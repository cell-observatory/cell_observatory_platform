import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data.io import read_zarr
from utils import cli
from utils.common import multiprocess
from data.datasets.pretrain_dataset import PretrainDataset
from data.data_shapes import  MULTICHANNEL_HYPERCUBE
from data.data_types import NUMPY_DTYPES, TENSORSTORE_DTYPES

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoadingBenchmark:
    def __init__(
        self,
        input_file: str|Path,
        outdir: str|Path,
        dtype: str = "fp16",
        zarr_file: bool = False,
        hypercube_shape: Optional[tuple] = None

    ):
        self.dtype = dtype
        self.input_file = Path(input_file)
        self.zarr_file = zarr_file

        if not zarr_file:
            self.dataset = PretrainDataset(
                hypercubes_dataframe_path=self.input_file,
                input_layout=MULTICHANNEL_HYPERCUBE.TZYXC,
                dtype=self.dtype,
                transforms=None,
                time=False
            )

        self.bytes_per_sample = np.dtype(NUMPY_DTYPES[self.dtype].value).itemsize

        if self.zarr_file and hypercube_shape:
            self.hypercube_shape = hypercube_shape
            self.channel_size = hypercube_shape[4]
            self.cube_size = hypercube_shape[2]  # assuming ZYX are same
            self.time_size = hypercube_shape[0]

            self.zarr_handle = read_zarr(str(self.input_file), dtype=self.dtype)
            zarr_shape = self.zarr_handle.shape
            logger.info(f"Zarr file shape: {zarr_shape}")
            self.max_time_start = max(0, zarr_shape[0] - self.time_size)
            self.max_z_start = max(0, zarr_shape[1] - self.cube_size)
            self.max_y_start = max(0, zarr_shape[2] - self.cube_size)
            self.max_x_start = max(0, zarr_shape[3] - self.cube_size)

        else:
            self.channel_size = self.dataset.hypercubes_dataframe.channel_size[0]
            self.cube_size = self.dataset.hypercubes_dataframe.cube_size[0]
            self.time_size = self.dataset.hypercubes_dataframe.time_size[0]
            self.hypercube_shape = (self.time_size, self.cube_size, self.cube_size, self.cube_size, self.channel_size)

        self.gb_per_hypercube = np.prod(self.hypercube_shape) * self.bytes_per_sample / (1024 ** 3)
        self.output_path = outdir / f"{"_".join(map(str, self.hypercube_shape))}_{self.dtype}"

        self.results = []

    def slice_hypercube(self, data_tensor, meta: Dict[str, Any]):
        t = slice(meta["time_start"], meta["time_start"] + meta["time_size"])
        c = slice(0, meta["channel_size"])
        z = slice(meta["z_start"], meta["z_start"] + meta["cube_size"])
        y = slice(meta["y_start"], meta["y_start"] + meta["cube_size"])
        x = slice(meta["x_start"], meta["x_start"] + meta["cube_size"])
        return data_tensor[t, z, y, x, c].read().result()

    def generate_hypercubes_grid(self):
        x_starts = np.arange(0, self.max_x_start + 1, self.cube_size)
        y_starts = np.arange(0, self.max_y_start + 1, self.cube_size)
        z_starts = np.arange(0, self.max_z_start + 1, self.cube_size)
        time_starts = np.arange(0, self.max_time_start + 1, self.time_size)

        hypercube_list = []
        for x in x_starts:
            for y in y_starts:
                for z in z_starts:
                    for t in time_starts:
                        hypercube_list.append(
                            {
                                'time_start': t,
                                'z_start': z,
                                'y_start': y,
                                'x_start': x,
                                'cube_size': self.cube_size,
                                'channel_size': self.channel_size,
                                'time_size': self.time_size,
                            }
                        )

        return hypercube_list

    def read_hypercube_from_zarr(self, rec) -> float:
        start = time.perf_counter()
        volume = self.slice_hypercube(self.zarr_handle, rec)
        read_time = time.perf_counter() - start
        return read_time, volume.nbytes

    def read_hypercube(self, rec):
        d = self.dataset._load_sample(rec, time_slice_hypercube=True)['meta']
        timer, nbytes = d['_slice_hypercube_timer'], d['_slice_hypercube_nbytes']
        return timer, nbytes

    def benchmark_parallel_reads(
        self,
        func: callable,
        hypercube_list: List[tuple],
        num_workers: int,
    ) -> Dict[str, float]:

        start_time = time.perf_counter()

        results = multiprocess(
            func=func,
            jobs=hypercube_list,
            cores=num_workers,
            desc=f"Loading hypercubes using {num_workers=} cpu(s)",
            unit='hypercube'
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
        logger.info(f"Starting benchmark for: {self.input_file}")
        if self.zarr_file:
            hypercube_list = self.generate_hypercubes_grid()[:num_hypercubes]
        else:
            hypercube_list = [self.dataset._index[i] for i in range(num_hypercubes)]

        for cpu_count in cpu_counts:
            logger.info(f"Benchmarking with {cpu_count} CPU cores...")


            if not self.zarr_file:
                [self.dataset.worker_init_fn(i) for i in range(cpu_count)]

            stats = self.benchmark_parallel_reads(
                hypercube_list=hypercube_list,
                num_workers=cpu_count,
                func=self.read_hypercube_from_zarr if self.zarr_file else self.read_hypercube
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


def main():
    parser = cli.argparser()
    parser.add_argument("ifile", help="Path to either zarr file or hypercubes dataframe csv")
    parser.add_argument("--hypercube-shape", nargs=5, type=int,
                       help="Hypercube shape as 5 integers: T Z Y X C (required when using --zarr-file)")
    parser.add_argument("--cpu-counts", nargs="+", type=int, default=list(range(1, 17)))
    parser.add_argument("--num-hypercubes", type=int, default=20)
    parser.add_argument("--dtype", type=str, default="fp16")

    args = parser.parse_args()

    ifile = Path(args.ifile)
    if not ifile.exists():
        logger.error(f"Path does not exist: {ifile}")
        sys.exit(1)

    outdir = Path(__file__).parent.parent / 'benchmarks'
    outdir.mkdir(parents=True, exist_ok=True)

    benchmarker = DataLoadingBenchmark(
        input_file=ifile,
        zarr_file=ifile.suffix == ".zarr",
        outdir=outdir,
        dtype=args.dtype,
        hypercube_shape=args.hypercube_shape
    )
    benchmarker.benchmark_cpu_scaling(
        cpu_counts=args.cpu_counts,
        num_hypercubes=args.num_hypercubes
    )

    print(f"\nBenchmark complete! Results saved to {outdir}")


if __name__ == "__main__":
    main()