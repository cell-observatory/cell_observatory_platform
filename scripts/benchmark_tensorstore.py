import sys
import time
from pathlib import Path
from typing import List, Dict
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import cli
from utils.common import multiprocess
from data.datasets.pretrain_dataset import PretrainDataset
from data.data_shapes import  MULTICHANNEL_HYPERCUBE
from data.data_types import  NUMPY_DTYPES


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoadingBenchmark:
    def __init__(
        self,
        hypercubes_dataframe_path: str|Path,
        outdir: str|Path,
        dtype: str = "fp16"
    ):
        self.dtype = dtype
        self.hypercubes_dataframe_path = Path(hypercubes_dataframe_path)
        self.dataset = PretrainDataset(
            hypercubes_dataframe_path=self.hypercubes_dataframe_path,
            input_layout=MULTICHANNEL_HYPERCUBE.TZYXC,
            dtype=self.dtype,
            transforms=None,
            time=False
        )

        self.bytes_per_sample = np.dtype(NUMPY_DTYPES[self.dtype].value).itemsize
        self.channel_size = self.dataset.hypercubes_dataframe.channel_size[0]
        self.cube_size = self.dataset.hypercubes_dataframe.cube_size[0]
        self.time_size = self.dataset.hypercubes_dataframe.time_size[0]
        self.hypercube_shape = (self.time_size, self.cube_size, self.cube_size, self.cube_size, self.channel_size)
        self.gb_per_hypercube = np.prod(self.hypercube_shape) * self.bytes_per_sample / (1024 ** 3)
        self.output_path = outdir / f"{"_".join(map(str, self.hypercube_shape))}_{self.dtype}.csv"

        self.results = []

    def read_hypercube(self, rec):
        d = self.dataset._load_sample(rec, time_slice_hypercube=True)['meta']
        timer, nbytes = d['_slice_hypercube_timer'], d['_slice_hypercube_nbytes']
        return timer, nbytes


    def benchmark_parallel_reads(self, hypercube_list: List[tuple], num_workers: int) -> Dict[str, float]:
        start_time = time.perf_counter()

        results = multiprocess(
            func=self.read_hypercube,
            jobs=hypercube_list,
            cores=num_workers,
            desc=f"Loading hypercubes using {num_workers=} cpu(s)",
            unit='hypercube'
        )

        total_time = time.perf_counter() - start_time

        individual_times = [r[0] for r in results]
        total_bytes = sum(r[1] for r in results)
        total_gb = total_bytes / (1024 ** 3)

        return {
            'total_time': total_time,
            'total_gb': total_gb,
            'throughput_gb_per_sec': total_gb / total_time,
            'throughput_gb_per_sec_per_core': (total_gb / total_time) / num_workers,
            'hypercubes_per_sec': len(hypercube_list) / total_time,
            'individual_times': individual_times,
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
            label=f'{self.hypercube_shape}, {self.dtype}'
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

        ax2.plot(cpu_cores, per_core_throughput, 's-', color='C1', linewidth=2.5, markersize=8, label='GB/s per Core')

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
        logger.info(f"Starting benchmark for: {self.hypercubes_dataframe_path}")
        hypercube_list = [self.dataset._index[i] for i in range(num_hypercubes)]

        for cpu_count in cpu_counts:
            logger.info(f"Benchmarking with {cpu_count} CPU cores...")
            [self.dataset.worker_init_fn(i) for i in range(cpu_count)]
            stats = self.benchmark_parallel_reads(hypercube_list, cpu_count)

            result = {
                'cpu_count': cpu_count,
                'num_hypercubes': num_hypercubes,
                'total_time': stats['total_time'],
                'total_gb': stats['total_gb'],
                'throughput_gb_per_sec': stats['throughput_gb_per_sec'],
                'throughput_gb_per_sec_per_core': stats['throughput_gb_per_sec_per_core'],
                'hypercubes_per_sec': stats['hypercubes_per_sec'],
                'mean_read_time': stats['mean_time'],
                'std_read_time': stats['std_time']
            }

            self.results.append(result)

            logger.info(
                f"CPU {cpu_count}: "
                f"{stats['throughput_gb_per_sec']:.2f} GB/s total, "
                f"{stats['throughput_gb_per_sec_per_core']:.3f} GB/s per core, "
                f"{stats['hypercubes_per_sec']:.2f} hypercubes/s, "
                f"({self.gb_per_hypercube:.3f} GB per hypercube)"
            )

            df = pd.DataFrame(self.results)
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.output_path, index=False)

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
    parser.add_argument("hypercubes_dataframe_path", help="Path to hypercubes dataframe csv")
    parser.add_argument("--cpu-counts", nargs="+", type=int, default=list(range(1, 17)))
    parser.add_argument("--num-hypercubes", type=int, default=20)
    parser.add_argument("--dtype", type=str, default="fp16")

    args = parser.parse_args()

    if not Path(args.hypercubes_dataframe_path).exists():
        logger.error(f"CSV path does not exist: {args.hypercubes_dataframe_path}")
        sys.exit(1)

    outdir = Path(__file__).parent.parent / 'benchmarks'
    outdir.mkdir(parents=True, exist_ok=True)

    benchmarker = DataLoadingBenchmark(args.hypercubes_dataframe_path, outdir=outdir, dtype=args.dtype)
    benchmarker.benchmark_cpu_scaling(
        cpu_counts=args.cpu_counts,
        num_hypercubes=args.num_hypercubes
    )

    print(f"\nBenchmark complete! Results saved to {outdir}")


if __name__ == "__main__":
    main()