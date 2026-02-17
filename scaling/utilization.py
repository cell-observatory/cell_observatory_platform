import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use('Agg')

import logging
import re
import sys
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import ujson
import wandb
from tqdm import tqdm

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_flops_profiler_log(log_text):
    """ Example:
        -------------------------- DeepSpeed Flops Profiler --------------------------
        Profile Summary at step 2:
        Notations:
        data parallel size (dp_size), model parallel size(mp_size),
        number of parameters (params), number of multiply-accumulate operations(MACs),
        number of floating-point operations (flops), floating-point operations per second (FLOPS),
        fwd latency (forward propagation latency), bwd latency (backward propagation latency),
        step (weights update latency), iter latency (sum of fwd, bwd and step latency)

        world size:                                                             8       
        data parallel size:                                                     8       
        model parallel size:                                                    1       
        batch size per GPU:                                                     288     
        params per GPU:                                                         189.37 M
        params of model = params per GPU * mp_size:                             189.37 M
        fwd MACs per GPU:                                                       90.14 TMACs
        fwd flops per GPU:                                                      180.32 T
        fwd flops of model = fwd flops per GPU * mp_size:                       180.32 T
        fwd latency:                                                            663.63 ms
        fwd FLOPS per GPU = fwd flops per GPU / fwd latency:                    271.72 TFLOPS
        bwd latency:                                                            889.99 ms
        bwd FLOPS per GPU = 2 * fwd flops per GPU / bwd latency:                405.22 TFLOPS
        fwd+bwd FLOPS per GPU = 3 * fwd flops per GPU / (fwd+bwd latency):      348.2 TFLOPS
        step latency:                                                           6.55 ms 
        iter latency:                                                           1.56 s  
        FLOPS per GPU = 3 * fwd flops per GPU / iter latency:                   346.73 TFLOPS
        samples/second:                                                         1476.77 
        ----------------------------- Aggregated Profile per GPU -----------------------------
        Top 1 modules in terms of params, MACs or fwd latency at different model depths:
        depth 0:
            params      - {'JEPA': '189.37 M'}
            MACs        - {'JEPA': '90.14 TMACs'}
            fwd latency - {'JEPA': '1.77 Gs'}
        depth 1:
            params      - {'MaskedEncoder': '182.66 M'}
            MACs        - {'MaskedEncoder': '84.89 TMACs'}
            fwd latency - {'MaskedEncoder': '569.37 ms'}
        depth 2:
            params      - {'Encoder': '176.39 M'}
            MACs        - {'Encoder': '82.63 TMACs'}
            fwd latency - {'Encoder': '592.4 ms'}
        depth 3:
            params      - {'ModuleList': '176.39 M'}
            MACs        - {'ModuleList': '82.63 TMACs'}
            fwd latency - {'ModuleList': '590.01 ms'}
        depth 4:
            params      - {'Transformer': '176.39 M'}
            MACs        - {'Transformer': '82.63 TMACs'}
            fwd latency - {'Transformer': '590.01 ms'}
        depth 5:
            params      - {'Mlp': '117.54 M'}
            MACs        - {'Attention': '44.2 TMACs'}
            fwd latency - {'Attention': '306.48 ms'}
    """
    deepspeed = {}
    pattern = re.compile(r"^(.*?):\s+([0-9\.,]+(?: [A-Za-z]+)?)\s*$", flags=re.MULTILINE)
    for match in pattern.finditer(log_text):
        key, value = match.groups()
        key = key.strip().replace(" ", "_").replace("=", "_eq_")
        parts = value.split()
        if len(parts) == 2:
            num_str, unit = parts
        else:
            num_str, unit = parts[0], ""
        try:
            num = float(num_str.replace(',', ''))
        except ValueError:
            num = num_str
        deepspeed[key] = {"value": num, "unit": unit} if unit else num
        
    top_modules = {}
    depth_block = re.compile(r"^depth\s+(\d+):\s*$", flags=re.MULTILINE)
    metric_line = re.compile(
        r"^\s+(params|MACs|fwd latency)\s+-\s+\{'([^']+)':\s+'([^']+)'\}\s*$",
        flags=re.MULTILINE,
    )
    for match in depth_block.finditer(log_text):
        depth = int(match.group(1))
        block_start = match.end()
        block_end = depth_block.search(log_text, block_start)
        block = log_text[block_start:block_end.start()] if block_end else log_text[block_start:]

        top_modules[depth] = {}

        for line_match in metric_line.finditer(block):
            metric, module_name, value_str = line_match.groups()
            metric = metric.replace(" ", "_")
            parts = value_str.split()
            top_modules[depth]["name"] = module_name.replace(" ", "_").replace("MaskedAutoEncoder", "MAE")
            

            if len(parts) >= 2:
                num_str, unit = parts[0], " ".join(parts[1:])
            else:
                num_str, unit = (parts[0], "") if parts else ("", "")
            try:
                value = float(num_str.replace(",", ""))
            except ValueError:
                value = num_str
            
            if metric == "params":
                top_modules[depth][metric] = convert_to_number(unit, value)
            elif metric == "MACs":
                top_modules[depth][metric] = convert_to_macs(unit, value)
            elif metric == "fwd_latency":
                top_modules[depth][metric] = convert_time_unit(unit, value)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
    return deepspeed, top_modules

def parse_wandb_hardware_log(wandb_filepath: Path):
    """ Example:
        {
        "os":  "Linux-5.14.0-611.5.1.el9_7.x86_64-x86_64-with-glibc2.39",
        "python":  "CPython 3.12.3",
        "startedAt":  "2026-02-05T19:25:12.695835Z",
        "args":  [
            "--node-ip-address=10.36.104.42",
            "--node-manager-port=19183",
            "--object-store-name=/tmp/symlink_b7dc6446f563/session_2026-02-05_14-23-29_551437_3592184/sockets/plasma_store",
            "--raylet-name=/tmp/symlink_b7dc6446f563/session_2026-02-05_14-23-29_551437_3592184/sockets/raylet",
            "--redis-address=None",
            "--metrics-agent-port=65412",
            "--logging-rotate-bytes=536870912",
            "--logging-rotate-backup-count=5",
            "--runtime-env-agent-port=47674",
            "--gcs-address=10.36.104.42:40651",
            "--session-name=session_2026-02-05_14-23-29_551437_3592184",
            "--temp-dir=/tmp/symlink_b7dc6446f563",
            "--webui=10.36.104.42:8265",
            "--cluster-id=3525789f94e74bd70f2e5920001367cbe79244cd73e7f23f16470412",
            "--startup-token=103",
            "--worker-launch-time-ms=1770319496223",
            "--node-id=96d47f4396514dc0a138d50a836972da792d3b5676114776ed49a644",
            "--runtime-env-hash=1239737103",
            "--enable-resource-isolation=false"
        ],
        "program":  "/usr/local/lib/python3.12/dist-packages/ray/_private/workers/default_worker.py",
        "root":  "/groups/betzig/betziglab/thayer/pretrained_models/pretraining/3D_JEPA-B_8xH200/logs",
        "host":  "h04u32.int.janelia.org",
        "executable":  "/usr/bin/python",
        "cpu_count":  96,
        "cpu_count_logical":  192,
        "gpu":  "NVIDIA H200",
        "gpu_count":  8,
        "disk":  {
            "/":  {
            "total":  "67108864",
            "used":  "16384"
            }
        },
        "memory":  {
            "total":  "4328097112064"
        },
        "gpu_nvidia":  [
            {
            "name":  "NVIDIA H200",
            "memoryTotal":  "150754820096",
            "cudaCores":  16896,
            "architecture":  "Hopper",
            "uuid":  "GPU-a3f71bf4-c9fa-d73d-3c74-1449be10b5c2"
            },
            {
            "name":  "NVIDIA H200",
            "memoryTotal":  "150754820096",
            "cudaCores":  16896,
            "architecture":  "Hopper",
            "uuid":  "GPU-2aaeb67c-9847-10e8-1ec8-e3e46403bead"
            },
            {
            "name":  "NVIDIA H200",
            "memoryTotal":  "150754820096",
            "cudaCores":  16896,
            "architecture":  "Hopper",
            "uuid":  "GPU-1ca9b5a9-7d32-47b0-1a91-f2604e204ff7"
            },
            {
            "name":  "NVIDIA H200",
            "memoryTotal":  "150754820096",
            "cudaCores":  16896,
            "architecture":  "Hopper",
            "uuid":  "GPU-bac34224-2f3e-982b-ce01-3086fdda2574"
            },
            {
            "name":  "NVIDIA H200",
            "memoryTotal":  "150754820096",
            "cudaCores":  16896,
            "architecture":  "Hopper",
            "uuid":  "GPU-9ad7176a-8013-78ff-5183-5c69880f435f"
            },
            {
            "name":  "NVIDIA H200",
            "memoryTotal":  "150754820096",
            "cudaCores":  16896,
            "architecture":  "Hopper",
            "uuid":  "GPU-e8f828ca-133f-07a6-79e2-456ac1cb4290"
            },
            {
            "name":  "NVIDIA H200",
            "memoryTotal":  "150754820096",
            "cudaCores":  16896,
            "architecture":  "Hopper",
            "uuid":  "GPU-232490ed-0f37-319a-3cb1-511be2ea79be"
            },
            {
            "name":  "NVIDIA H200",
            "memoryTotal":  "150754820096",
            "cudaCores":  16896,
            "architecture":  "Hopper",
            "uuid":  "GPU-3e0a1691-6474-0cc4-3513-0cd8c73befc1"
            }
        ],
        "cudaVersion":  "13.1",
        "writerId":  "b3ce1mlon9cafqbvd99laiatgrtnbgvh"
        }
    """
    with (wandb_filepath).open() as f:
        hardware_log = ujson.load(f)
        
    return {
        "gpu": hardware_log["gpu"].replace("NVIDIA ", ""),
        "gpu_count": hardware_log["gpu_count"],
        "physical_cpu_cores": hardware_log["cpu_count"],
        "logical_cpu_cores": hardware_log["cpu_count_logical"],
        "ram_total": hardware_log["memory"]["total"],
        "vram_total": hardware_log["gpu_nvidia"][0]["memoryTotal"],
        "cuda_cores": hardware_log["gpu_nvidia"][0]["cudaCores"],
        "cudaVersion": hardware_log["cudaVersion"],
        "host": hardware_log["host"],
        "os": hardware_log["os"],
        "python": hardware_log["python"],
    }

def convert_to_number(unit: str, value: float):
    if unit == "M":
        return value * 1e6
    elif unit == "B" or unit == "G":
        return value * 1e9
    elif unit == "T":
        return value * 1e12
    elif unit == "P":
        return value * 1e15
    elif unit == "E":
        return value * 1e18
    else:
        raise ValueError(f"Unknown unit: {unit}")

def convert_to_flops(unit: str, value: float):
    if unit == "MFLOPS":
        return value * 1e6 
    elif unit == "GFLOPS":
        return value * 1e9
    elif unit == "TFLOPS":
        return value * 1e12
    elif unit == "PFLOPS":
        return value * 1e15
    elif unit == "EFLOPS":
        return value * 1e18
    else:
        raise ValueError(f"Unknown unit: {unit}")

def convert_to_macs(unit: str, value: float):
    if unit == "MMACs":
        return value * 1e6
    elif unit == "TMACs":
        return value * 1e12
    elif unit == "GMACs":
        return value * 1e9
    else:
        raise ValueError(f"Unknown unit: {unit}")

def convert_time_unit(unit: str, value: float):
    if unit == "ms":
        return value / 1e3
    elif unit == "us":
        return value / 1e6
    elif unit == "ns":
        return value / 1e9
    elif unit == "s":
        return value
    elif unit == "Gs":
        return value * 1e9
    else:
        raise ValueError(f"Unknown unit: {unit}")
    
def get_model_params(flops_profiler: dict):
    log = flops_profiler["params_of_model__eq__params_per_GPU_*_mp_size"]
    unit, value = log["unit"], log["value"]
    return convert_to_number(unit, value)
    
def get_step_latency(flops_profiler: dict):
    log = flops_profiler["step_latency"]
    unit, value = log["unit"], log["value"]
    return convert_time_unit(unit, value)

def get_iter_latency(flops_profiler: dict):
    
    log = flops_profiler["iter_latency"]
    unit, value = log["unit"], log["value"]
    return convert_time_unit(unit, value)

def get_fwd_latency(flops_profiler: dict):
    log = flops_profiler["fwd_latency"]
    unit, value = log["unit"], log["value"]
    return convert_time_unit(unit, value)

def get_bwd_latency(flops_profiler: dict):
    log = flops_profiler["bwd_latency"]
    unit, value = log["unit"], log["value"]
    return convert_time_unit(unit, value)

def get_fwd_flops(flops_profiler: dict):
    log = flops_profiler["fwd_flops_of_model__eq__fwd_flops_per_GPU_*_mp_size"]
    unit, value = log["unit"], log["value"]
    return convert_to_number(unit, value)

def get_fwd_flops_per_gpu(flops_profiler: dict):
    log = flops_profiler["fwd_flops_per_GPU"]
    unit, value = log["unit"], log["value"]
    return convert_to_number(unit, value)

def get_bwd_flop_per_second_per_gpu(flops_profiler: dict):
    log = flops_profiler["bwd_FLOPS_per_GPU__eq__2_*_fwd_flops_per_GPU_/_bwd_latency"]
    unit, value = log["unit"], log["value"]
    return convert_to_flops(unit, value)

def get_fwd_flop_per_second_per_gpu(flops_profiler: dict):
    log = flops_profiler["fwd_FLOPS_per_GPU__eq__fwd_flops_per_GPU_/_fwd_latency"]
    unit, value = log["unit"], log["value"]
    return convert_to_flops(unit, value)

def get_flop_per_second_per_gpu(flops_profiler: dict):
    log = flops_profiler["FLOPS_per_GPU__eq__3_*_fwd_flops_per_GPU_/_iter_latency"]
    unit, value = log["unit"], log["value"]
    return convert_to_flops(unit, value)

def get_pretraining_token_size(token_shape: dict, token_channel_dtype: str):
    if token_channel_dtype == "fp16":
        return np.prod(list(token_shape.values())) * 2 / 1024**3 # GiB
    elif token_channel_dtype == "fp32":
        return np.prod(list(token_shape.values())) * 4 / 1024**3 # GiB
    else:
        raise ValueError(f"Unknown channel dtype: {token_channel_dtype}")

def get_utilization(
    datadir: Path, 
    wandb_project: str = "profiling",
    token_shape: dict = {'t': 128, 'z': 256, 'y': 256, 'x': 2}, 
    token_channel_dtype: str = "fp16",
    outdir: Path = Path("../utilization/data"),
):
    models = {}
    for model_dir in tqdm(datadir.glob("*/")):
        
        scalars_logsdir = model_dir / "logs" / "scalars"
        flops_logdir = model_dir / "logs" / model_dir.name
        wandb_logdir = model_dir / "logs" / "wandb"
        
        if not scalars_logsdir.exists():
            continue
        

        print(f"Processing {model_dir.name}")
        epoch_logbook = pd.read_csv(scalars_logsdir / "epoch_logbook.csv")
        step_logbook = pd.read_csv(scalars_logsdir / "step_logbook.csv")
        
        with (flops_logdir / "flops_profiler.log").open() as f:
            log_text = f.read()
            flops_profiler, top_modules = parse_flops_profiler_log(log_text)
            training_gpus = int(flops_profiler["world_size"])
            batch_size_per_gpu = int(flops_profiler["batch_size_per_GPU"])
            # pprint(flops_profiler)
            # pprint(top_modules)
        
        token_size = get_pretraining_token_size(token_shape, token_channel_dtype)
        hardware_specs = parse_wandb_hardware_log(wandb_logdir.rglob("wandb-metadata.json").__next__())
        
        print(f"Looking for wandb logs for {model_dir.name}")
        runid = wandb_logdir.rglob("run-*.wandb").__next__().stem.replace("run-", "")
        run = wandb.Api().run(f"cell-observatory/{wandb_project}/{model_dir.name}/{runid}")
        run_stats = run.history(stream="system") 
        training_gpus_per_node = hardware_specs["gpu_count"]
        
        run_stats["system.gpu.gpu"] = run_stats[[f"system.gpu.{gpu_id}.gpu" for gpu_id in range(training_gpus_per_node)]].mean(axis=1)
        run_stats["system.gpu.memoryAllocatedBytes"] = run_stats[[f"system.gpu.{gpu_id}.memoryAllocatedBytes" for gpu_id in range(training_gpus_per_node)]].mean(axis=1)
        run_stats["system.gpu.memoryAllocated"] = run_stats[[f"system.gpu.{gpu_id}.memoryAllocated" for gpu_id in range(training_gpus_per_node)]].mean(axis=1)
        run_stats["system.gpu.temp"] = run_stats[[f"system.gpu.{gpu_id}.temp" for gpu_id in range(training_gpus_per_node)]].mean(axis=1)
        run_stats["system.gpu.powerWatts"] = run_stats[[f"system.gpu.{gpu_id}.powerWatts" for gpu_id in range(training_gpus_per_node)]].mean(axis=1)
        
        avg_vram_per_gpu = run_stats[run_stats["system.gpu.memoryAllocatedBytes"] > 0]["system.gpu.memoryAllocatedBytes"].mean() / 1e9
        avg_vram_percent_per_gpu = run_stats[run_stats["system.gpu.memoryAllocated"] > 0]["system.gpu.memoryAllocated"].mean()
        avg_temp_per_gpu = run_stats[run_stats["system.gpu.temp"] > 0]["system.gpu.temp"].mean()
        avg_power_per_gpu = run_stats[run_stats["system.gpu.powerWatts"] > 0]["system.gpu.powerWatts"].mean()
        avg_utilization_per_gpu = run_stats[run_stats["system.gpu.gpu"] > 0]["system.gpu.gpu"].mean()
        avg_disk_utilization = run_stats[run_stats["system.disk./.usagePercent"] > 0]["system.disk./.usagePercent"].mean()
        avg_ram_utilization = run_stats[run_stats["system.memory_percent"] > 0]["system.memory_percent"].mean()
        avg_cpu_threads = run_stats[run_stats["system.proc.cpu.threads"] > 0]["system.proc.cpu.threads"].mean()
        
        q75_vram_per_gpu = run_stats[run_stats["system.gpu.memoryAllocatedBytes"] > 0]["system.gpu.memoryAllocatedBytes"].quantile(0.75) / 1e9
        q75_vram_percent_per_gpu = run_stats[run_stats["system.gpu.memoryAllocated"] > 0]["system.gpu.memoryAllocated"].quantile(0.75)
        q75_temp_per_gpu = run_stats[run_stats["system.gpu.temp"] > 0]["system.gpu.temp"].quantile(0.75)
        q75_power_per_gpu = run_stats[run_stats["system.gpu.powerWatts"] > 0]["system.gpu.powerWatts"].quantile(0.75)
        q75_utilization_per_gpu = run_stats[run_stats["system.gpu.gpu"] > 0]["system.gpu.gpu"].quantile(0.75)
        q75_disk_utilization = run_stats[run_stats["system.disk./.usagePercent"] > 0]["system.disk./.usagePercent"].quantile(0.75)
        q75_ram_utilization = run_stats[run_stats["system.memory_percent"] > 0]["system.memory_percent"].quantile(0.75)
        q75_cpu_threads = run_stats[run_stats["system.proc.cpu.threads"] > 0]["system.proc.cpu.threads"].quantile(0.75)
        
        max_vram_per_gpu = run_stats["system.gpu.memoryAllocatedBytes"].max() / 1e9
        max_vram_percent_per_gpu = run_stats["system.gpu.memoryAllocated"].max()
        max_temp_per_gpu = run_stats["system.gpu.temp"].max()
        max_power_per_gpu = run_stats["system.gpu.powerWatts"].max()
        max_utilization_per_gpu = run_stats["system.gpu.gpu"].max()
        max_disk_utilization = run_stats["system.disk./.usagePercent"].max()
        max_ram_utilization = run_stats["system.memory_percent"].max()
        max_cpu_threads = run_stats["system.proc.cpu.threads"].max()
        
        
        steps_per_epoch = step_logbook.groupby("epoch").size().max()
        tokens_per_epoch = batch_size_per_gpu * training_gpus * steps_per_epoch
        gib_per_epoch = tokens_per_epoch * token_size / 1024**3
        
        models[model_dir.name] = {
            "training_gpus": training_gpus,
            "batch_size_per_gpu": batch_size_per_gpu,
            "model_parallel_size": int(flops_profiler["model_parallel_size"]),
            "batch_size": training_gpus * batch_size_per_gpu,
            "steps_per_epoch": steps_per_epoch,
            "model_params": get_model_params(flops_profiler),
            "epoch_time": epoch_logbook["epoch_time_median"].mean(),
            "epoch_data_time": epoch_logbook["data_time_median"].mean(),
            "epoch_masking_time": epoch_logbook["masking_time_median"].mean(),
            "epoch_preprocess_time": epoch_logbook["preprocess_time_median"].mean(),
            "avg_vram": epoch_logbook["max_allocated_mem_median"].mean(),
            "step_time": step_logbook["step_time_median"].mean(),
            "step_data_time": step_logbook["data_time_median"].mean(),
            "step_masking_time": step_logbook["masking_time_median"].mean(),
            "step_preprocess_time": step_logbook["preprocess_time_median"].mean(),
            "step_latency": get_step_latency(flops_profiler),
            "iter_latency": get_iter_latency(flops_profiler),
            "fwd_latency": get_fwd_latency(flops_profiler),
            "bwd_latency": get_bwd_latency(flops_profiler),
            "fwd_flops": get_fwd_flops(flops_profiler),
            "fwd_flops_per_gpu": get_fwd_flops_per_gpu(flops_profiler),
            "fwd_flop_per_second_per_gpu": get_fwd_flop_per_second_per_gpu(flops_profiler),
            "bwd_flop_per_second_per_gpu": get_bwd_flop_per_second_per_gpu(flops_profiler),
            "flop_per_second_per_gpu": get_flop_per_second_per_gpu(flops_profiler),
            "tokens_per_second": int(flops_profiler["samples/second"]),
            "tokens_per_second_per_gpu": int(flops_profiler["samples/second"]) / training_gpus,
            "gib_per_second": int(flops_profiler["samples/second"]) * token_size,
            "gib_per_second_per_gpu": int(flops_profiler["samples/second"]) * token_size / training_gpus,
            "gib_per_batch_per_gpu": batch_size_per_gpu * token_size,
            "avg_temp_per_gpu": avg_temp_per_gpu,
            "avg_vram_per_gpu": avg_vram_per_gpu,
            "avg_vram_percent_per_gpu": avg_vram_percent_per_gpu,
            "avg_power_per_gpu": avg_power_per_gpu,
            "avg_utilization_per_gpu": avg_utilization_per_gpu,
            "avg_disk_utilization": avg_disk_utilization,
            "avg_ram_utilization": avg_ram_utilization,
            "avg_cpu_threads": avg_cpu_threads,
            "q75_vram_per_gpu": q75_vram_per_gpu,
            "q75_vram_percent_per_gpu": q75_vram_percent_per_gpu,
            "q75_temp_per_gpu": q75_temp_per_gpu,
            "q75_power_per_gpu": q75_power_per_gpu,
            "q75_utilization_per_gpu": q75_utilization_per_gpu,
            "q75_disk_utilization": q75_disk_utilization,
            "q75_ram_utilization": q75_ram_utilization,
            "q75_cpu_threads": q75_cpu_threads,
            "tokens_per_epoch": tokens_per_epoch,
            "max_vram_per_gpu": max_vram_per_gpu,
            "max_vram_percent_per_gpu": max_vram_percent_per_gpu,
            "max_temp_per_gpu": max_temp_per_gpu,
            "max_power_per_gpu": max_power_per_gpu,
            "max_utilization_per_gpu": max_utilization_per_gpu,
            "max_disk_utilization": max_disk_utilization,
            "max_ram_utilization": max_ram_utilization,
            "max_cpu_threads": max_cpu_threads,
            "gib_per_epoch": gib_per_epoch,
            "token_shape": token_shape,
            **hardware_specs
        }
        
        for depth in top_modules.keys():
            models[model_dir.name][f"main_module_{depth}_name"] = top_modules[depth]["name"]
            models[model_dir.name][f"main_module_{depth}_params"] = top_modules[depth]["params"]
            models[model_dir.name][f"main_module_{depth}_MACs"] = top_modules[depth]["MACs"]
            models[model_dir.name][f"main_module_{depth}_fwd_latency"] = top_modules[depth]["fwd_latency"]
        
        
    models = pd.DataFrame.from_dict(models, orient="index").reset_index(drop=False)
    print(models)
    print(models.columns)
    models.to_csv(outdir / "utilization.csv", index=False)
    return models