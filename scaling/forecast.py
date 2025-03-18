import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from pathlib import Path

import logging
import sys
import time

from scaling import vis
from scaling import profile
from utils import cli

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args(args):
    parser = cli.argparser()

    parser.add_argument(
        "--ishape", default={'t': 16, 'z': 128, 'y': 128, 'x': 128, 'c': 3}, type=dict,
    )

    parser.add_argument(
        "--ipatch", default={'t': 2, 'z': 16, 'y': 16, 'x': 16, 'c': 3}, type=dict,
    )

    parser.add_argument("--rgb", action="store_true")

    parser.add_argument(
        "--outdir", default="../scaling/data", type=Path, help='path to save trained models'
    )

    parser.add_argument(
        "--arch", type=str, default='vit', choices=["published_models", "vit", "mae_ssl", "mae_ft", "transformer"],
        help='architecture to use'
    )

    parser.add_argument(
        "--mask_ratio", type=float, default=0.0,
        help='mask ratio for mae pretraining'
    )

    parser.add_argument(
        "--cost_h100_per_hr", type=float, default=49.24/8,
        help='https://www.coreweave.com/pricing'
    )

    parser.add_argument(
        "--dtype", type=str, default='float16', choices=["float8", "float16", "float32", "float64"],
        help='optional dtype to use'
    )

    parser.add_argument(
        "--batch_size", default=4096, type=int, help="number of volumes per batch"
    )

    return parser.parse_known_args(args)[0]


def scaling_transformer(
    ishape={'t': 16, 'z': 128, 'y': 128, 'x': 128, 'c': 3},
    dtype='float16',
    outdir=Path("../scaling/data/transformers")
):
    dimensions = {
        "2D(g)": {"t": 1, "z": 1, "y": ishape['y'], "x": ishape['x'], "c": 1},
        "2D(rgb)": {"t": 1, "z": 1, "y": ishape['y'], "x": ishape['x'], "c": ishape['c']},
        "3D(g)": {"t": 1, "z": ishape['z'], "y": ishape['y'], "x": ishape['x'], "c": 1},
        "3D(rgb)": {"t": 1, "z": ishape['z'], "y": ishape['y'], "x": ishape['x'], "c": ishape['c']},
        "4D(g)": {"t": ishape['t'], "z": ishape['z'], "y": ishape['y'], "x": ishape['x'], "c": 1},
        "4D(rgb)": {"t": ishape['t'], "z": ishape['z'], "y": ishape['y'], "x": ishape['x'], "c": ishape['c']}
    }
    configs = {
        "1b": {"layers": 1, "heads": 16, "embedding": 128, "mlp": 512},
        "2b": {"layers": 2, "heads": 16, "embedding": 256, "mlp": 1024},
        "4b": {"layers": 4, "heads": 16, "embedding": 512, "mlp": 2048},
        "8b": {"layers": 8, "heads": 16, "embedding": 768, "mlp": 3072},
        "T": {"layers": 12, "heads": 3, "embedding": 192, "mlp": 768},
        "S": {"layers": 12, "heads": 6, "embedding": 384, "mlp": 1536},
        "B": {"layers": 12, "heads": 12, "embedding": 768, "mlp": 3072},
        "L": {"layers": 24, "heads": 16, "embedding": 1024, "mlp": 4096},
        "H": {"layers": 32, "heads": 16, "embedding": 1280, "mlp": 5120},
        "g": {"layers": 40, "heads": 16, "embedding": 1408, "mlp": 6144},
        "G": {"layers": 48, "heads": 16, "embedding": 1664, "mlp": 8192},
        "e": {"layers": 56, "heads": 16, "embedding": 1792, "mlp": 15360},
        "22B": {"layers": 48, "heads": 48, "embedding": 6144, "mlp": 24576},
    }

    transformer_configs, vit_configs = {}, {}
    for patch in [14, 16]:
        patches = {
            "2D(g)": {"t": 1, "z": 1, "y": patch, "x": patch, "c": 1},
            "2D(rgb)": {"t": 1, "z": 1, "y": patch, "x": patch, "c": 3},
            "3D(g)": {"t": 1, "z": patch, "y": patch, "x": patch, "c": 1},
            "3D(rgb)": {"t": 1, "z": patch, "y": patch, "x": patch, "c": 3},
            "4D(g)": {"t": 2, "z": patch, "y": patch, "x": patch, "c": 1},
            "4D(rgb)": {"t": 2, "z": patch, "y": patch, "x": patch, "c": 3}
        }

        for dims in dimensions.keys():

            volume_size = list(dimensions[dims].values())
            patch_size = list(patches[dims].values())
            num_tokens = profile.patchify(volume_size=volume_size, patch_size=patch_size)

            memory_per_volume = profile.data_memory_footprint(
                volume_size=volume_size,
                batch_size=1,
                dtype=dtype,
            )

            for transformer in ["En", "De", "AE"]:
                for c in configs:
                    print(f"{dims} {c}/{patch} {transformer}")
                    layers = configs[c]["layers"]
                    heads = configs[c]["heads"]
                    embedding = configs[c]["embedding"]
                    mlp = configs[c]["mlp"]

                    if transformer == "En":
                        params = profile.encoder_transformer_params(
                            layers=layers,
                            embed_dim=embedding,
                            mlp_dim=mlp
                        )
                        flops = profile.encoder_transformer_flops(
                            num_tokens=num_tokens,
                            layers=layers,
                            embed_dim=embedding,
                            heads=heads,
                            mlp_dim=mlp
                        )
                        flops_per_token = layers * profile.encoder_flops(1, embedding, heads, mlp)
                    elif transformer == "De":
                        params = profile.decoder_transformer_params(
                            layers=layers,
                            embed_dim=embedding,
                            mlp_dim=mlp
                        )
                        flops = profile.decoder_transformer_flops(
                            num_tokens=num_tokens,
                            layers=layers,
                            embed_dim=embedding,
                            heads=heads,
                            mlp_dim=mlp
                        )
                        flops_per_token = layers * profile.decoder_flops(1, embedding, heads, mlp)
                    else:
                        eparams = profile.encoder_transformer_params(
                            layers=layers,
                            embed_dim=embedding,
                            mlp_dim=mlp
                        )
                        dparams = profile.decoder_transformer_params(
                            layers=layers,
                            embed_dim=embedding,
                            mlp_dim=mlp
                        )
                        params = eparams + dparams

                        eflops = profile.encoder_transformer_flops(
                            num_tokens=num_tokens,
                            layers=layers,
                            embed_dim=embedding,
                            heads=heads,
                            mlp_dim=mlp
                        )
                        eflops_per_token = layers * profile.encoder_flops(1, embedding, heads, mlp)

                        dflops = profile.decoder_transformer_flops(
                            num_tokens=num_tokens,
                            layers=layers,
                            embed_dim=embedding,
                            heads=heads,
                            mlp_dim=mlp
                        )
                        dflops_per_token = layers * profile.decoder_flops(1, embedding, heads, mlp)

                        flops = eflops + dflops
                        flops_per_token = eflops_per_token + dflops_per_token

                    gflops = np.round(flops / 1e9, 3)
                    gflops_per_patch = np.round(flops_per_token / 1e9, 3)

                    model_inference_memory = profile.transformer_inference_memory_footprint(
                        params=params,
                        dtype=dtype
                    )

                    model_training_memory = profile.transformer_training_memory_footprint(
                        params=params,
                        dtype=dtype
                    )

                    inference_time_per_volume = profile.compute_time(flops=flops, gpu="H100", unit="seconds")
                    training_time_per_volume = profile.compute_time(flops=3 * flops, gpu="H100", unit="seconds")

                    patches_per_volume = np.product([s // p for s, p in zip(volume_size, patch_size)])
                    pixels_per_patch = np.product(patch_size)
                    volumes_per_h100 = 80 // memory_per_volume

                    transformer_configs[f"{dims} {c}/{patch} {transformer}"] = {
                        "data": dims,
                        "class": f"{c}/{patch}",
                        "transformer": transformer,
                        "layers": layers,
                        "heads": heads,
                        "mlp": mlp,
                        "embedding": embedding,
                        "t": dimensions[dims]["t"],
                        "x": dimensions[dims]["x"],
                        "y": dimensions[dims]["y"],
                        "z": dimensions[dims]["z"],
                        "c": dimensions[dims]["c"],
                        "pt": patches[dims]["t"],
                        "px": patches[dims]["x"],
                        "py": patches[dims]["y"],
                        "pz": patches[dims]["z"],
                        "pc": patches[dims]["c"],
                        "patches_per_volume": patches_per_volume,
                        "pixels_per_patch": pixels_per_patch,
                        "volumes_per_h100": volumes_per_h100,
                        "memory_per_volume": memory_per_volume,
                        "parameters": params,
                        "inference_gflops_per_volume": gflops,
                        "training_gflops_per_volume": 3 * gflops,
                        "gflops_per_patch": gflops_per_patch,
                        "model_inference_memory": model_inference_memory,
                        "model_training_memory": model_training_memory,
                        "inference_time_per_volume": inference_time_per_volume,
                        "training_time_per_volume": training_time_per_volume,
                    }

    transformer_scaling = pd.DataFrame.from_dict(transformer_configs, orient='index')
    transformer_scaling = transformer_scaling.sort_values(['px', 'parameters', 'layers', 'heads'],
                                                          ascending=[True, True, True, True])
    transformer_scaling.to_csv(outdir / "transformers.csv")
    return transformer_scaling


def scaling_vit(
    ishape={'t': 16, 'z': 128, 'y': 128, 'x': 128, 'c': 3},
    dtype='float16',  # per channel
    outdir=Path("../scaling/data/vits")
):
    vit_dimensions = {
        "2D(g)": {"t": 1, "z": 1, "y": ishape['y'], "x": ishape['x'], "c": 1},
        "2D(rgb)": {"t": 1, "z": 1, "y": ishape['y'], "x": ishape['x'], "c": ishape['c']},
        "3D(g)": {"t": 1, "z": ishape['z'], "y": ishape['y'], "x": ishape['x'], "c": 1},
        "3D(rgb)": {"t": 1, "z": ishape['z'], "y": ishape['y'], "x": ishape['x'], "c": ishape['c']},
        "4D(g)": {"t": ishape['t'], "z": ishape['z'], "y": ishape['y'], "x": ishape['x'], "c": 1},
        "4D(rgb)": {"t": ishape['t'], "z": ishape['z'], "y": ishape['y'], "x": ishape['x'], "c": ishape['c']}
    }
    vits = {
        "S": {"layers": 12, "heads": 6, "embedding": 384, "mlp": 1536},
        "B": {"layers": 12, "heads": 12, "embedding": 768, "mlp": 3072},
        "L": {"layers": 24, "heads": 16, "embedding": 1024, "mlp": 4096},
        "H": {"layers": 32, "heads": 16, "embedding": 1280, "mlp": 5120},
        "g": {"layers": 40, "heads": 16, "embedding": 1408, "mlp": 6144},
        "G": {"layers": 48, "heads": 16, "embedding": 1664, "mlp": 8192},
        "e": {"layers": 56, "heads": 16, "embedding": 1792, "mlp": 15360},
        "22B": {"layers": 48, "heads": 48, "embedding": 6144, "mlp": 24576},
    }

    vit_configs = {}
    for patch in [14, 16]:
        patches = {
            "2D(g)": {"t": 1, "z": 1, "y": patch, "x": patch, "c": 1},
            "2D(rgb)": {"t": 1, "z": 1, "y": patch, "x": patch, "c": 3},
            "3D(g)": {"t": 1, "z": patch, "y": patch, "x": patch, "c": 1},
            "3D(rgb)": {"t": 1, "z": patch, "y": patch, "x": patch, "c": 3},
            "4D(g)": {"t": 2, "z": patch, "y": patch, "x": patch, "c": 1},
            "4D(rgb)": {"t": 2, "z": patch, "y": patch, "x": patch, "c": 3}
        }

        for dims in vit_dimensions.keys():

            volume_size = list(vit_dimensions[dims].values())
            patch_size = list(patches[dims].values())
            num_tokens = profile.patchify(volume_size=volume_size, patch_size=patch_size)

            memory_per_volume = profile.data_memory_footprint(
                volume_size=volume_size,
                batch_size=1,
                dtype=dtype,
            )

            for v in vits:
                print(f"{dims} ViT {v}/{patch}")

                layers = vits[v]["layers"]
                heads = vits[v]["heads"]
                embedding = vits[v]["embedding"]
                mlp = vits[v]["mlp"]

                params = profile.encoder_transformer_params(
                    layers=layers,
                    embed_dim=embedding,
                    mlp_dim=mlp
                )
                flops = profile.encoder_transformer_flops(
                    num_tokens=num_tokens,
                    layers=layers,
                    embed_dim=embedding,
                    heads=heads,
                    mlp_dim=mlp
                )
                flops_per_patch = layers * profile.encoder_flops(1, embedding, heads, mlp)

                model_inference_memory = profile.transformer_inference_memory_footprint(
                    params=params,
                    dtype=dtype
                )

                model_training_memory = profile.transformer_training_memory_footprint(
                    params=params,
                    dtype=dtype
                )

                inference_time_per_volume = profile.compute_time(flops=flops, gpu="H100", unit="seconds")
                training_time_per_volume = profile.compute_time(flops=3 * flops, gpu="H100", unit="seconds")
                gflops = np.round(flops / 1e9, 3)
                gflops_per_patch = np.round(flops_per_patch / 1e9, 3)

                patches_per_volume = np.product([s // p for s, p in zip(volume_size, patch_size)])
                pixels_per_patch = np.product(patch_size)
                volumes_per_h100 = 80 // memory_per_volume

                """
                    ViT L/16: https://arxiv.org/pdf/2010.11929.pdf (table 6)
                    exaFLOPs = 783
                    epochs = 7
                    dataset = 303,000,000
                    TPUv3 peak FLOPS = 123 * 10**12

                    training_time_per_volume = (783 * 10^18) / 7 / 303,000,000 / (123 * 10**12)
                    training_time_per_volume = 0.00300134543
                """
                vit_configs[f"{dims} ViT {v}/{patch}"] = {
                    "data": dims,
                    "class": f"{v}/{patch}",
                    "transformer": "encoder",
                    "layers": layers,
                    "heads": heads,
                    "mlp": mlp,
                    "embedding": embedding,
                    "t": vit_dimensions[dims]["t"],
                    "x": vit_dimensions[dims]["x"],
                    "y": vit_dimensions[dims]["y"],
                    "z": vit_dimensions[dims]["z"],
                    "c": vit_dimensions[dims]["c"],
                    "pt": patches[dims]["t"],
                    "px": patches[dims]["x"],
                    "py": patches[dims]["y"],
                    "pz": patches[dims]["z"],
                    "pc": patches[dims]["c"],
                    "patches_per_volume": patches_per_volume,
                    "pixels_per_patch": pixels_per_patch,
                    "volumes_per_h100": volumes_per_h100,
                    "memory_per_volume": memory_per_volume,
                    "parameters": params,
                    "inference_gflops_per_volume": gflops,
                    "training_gflops_per_volume": 3 * gflops,
                    "gflops_per_patch": gflops_per_patch,
                    "model_inference_memory": model_inference_memory,
                    "model_training_memory": model_training_memory,
                    "inference_time_per_volume": inference_time_per_volume,
                    "training_time_per_volume": training_time_per_volume,
                }

    vit_scaling = pd.DataFrame.from_dict(vit_configs, orient='index')
    vit_scaling = vit_scaling.sort_values(['px', 'parameters', 'layers', 'heads'], ascending=[True, True, True, True])
    vit_scaling.to_csv(outdir / "vits.csv")
    return vit_scaling


def scaling_mae_ssl(
    ishape={'t': 16, 'z': 128, 'y': 128, 'x': 128, 'c': 3},
    dtype='float16',  # per channel
    outdir=Path("../scaling/data/maes/ssl"),
    mask_ratio=0.75
):
    maes_dimensions = {
        "2D(g)": {"t": 1, "z": 1, "y": ishape['y'], "x": ishape['x'], "c": 1},
        "2D(rgb)": {"t": 1, "z": 1, "y": ishape['y'], "x": ishape['x'], "c": ishape['c']},
        "3D(g)": {"t": 1, "z": ishape['z'], "y": ishape['y'], "x": ishape['x'], "c": 1},
        "3D(rgb)": {"t": 1, "z": ishape['z'], "y": ishape['y'], "x": ishape['x'], "c": ishape['c']},
        "4D(g)": {"t": ishape['t'], "z": ishape['z'], "y": ishape['y'], "x": ishape['x'], "c": 1},
        "4D(rgb)": {"t": ishape['t'], "z": ishape['z'], "y": ishape['y'], "x": ishape['x'], "c": ishape['c']}
    }
    maes_encoders = {
        "S": {"layers": 12, "heads": 6, "embedding": 384, "mlp": 1536},
        "B": {"layers": 12, "heads": 12, "embedding": 768, "mlp": 3072},
        "L": {"layers": 24, "heads": 16, "embedding": 1024, "mlp": 4096},
        "H": {"layers": 32, "heads": 16, "embedding": 1280, "mlp": 5120},
        "2B": {"layers": 24, "heads": 32, "embedding": 2560, "mlp": 10240},
        "6B": {"layers": 32, "heads": 32, "embedding": 4096, "mlp": 16384},
        "g": {"layers": 40, "heads": 16, "embedding": 1408, "mlp": 6144},
        "G": {"layers": 48, "heads": 16, "embedding": 1664, "mlp": 8192},
        "e": {"layers": 56, "heads": 16, "embedding": 1792, "mlp": 15360},
        "22B": {"layers": 48, "heads": 48, "embedding": 6144, "mlp": 24576},
    }
    maes_decoders = {
        "B": {"layers": 8, "heads": 16, "embedding": 512, "mlp": 2048},
    }

    mae_configs = {}
    for patch in [14, 16]:
        patches = {
            "2D(g)": {"t": 1, "z": 1, "y": patch, "x": patch, "c": 1},
            "2D(rgb)": {"t": 1, "z": 1, "y": patch, "x": patch, "c": 3},
            "3D(g)": {"t": 1, "z": patch, "y": patch, "x": patch, "c": 1},
            "3D(rgb)": {"t": 1, "z": patch, "y": patch, "x": patch, "c": 3},
            "4D(g)": {"t": 2, "z": patch, "y": patch, "x": patch, "c": 1},
            "4D(rgb)": {"t": 2, "z": patch, "y": patch, "x": patch, "c": 3}
        }

        for dims in maes_dimensions.keys():

            volume_size = list(maes_dimensions[dims].values())
            patch_size = list(patches[dims].values())
            num_encoder_tokens = profile.patchify(volume_size=volume_size, patch_size=patch_size, mask_ratio=mask_ratio)
            num_decoder_tokens = profile.patchify(volume_size=volume_size, patch_size=patch_size, mask_ratio=0.0)

            memory_per_volume = profile.data_memory_footprint(
                volume_size=volume_size,
                batch_size=1,
                dtype=dtype,
            )

            for v in maes_encoders:
                print(f"{dims} MAE {v}/{patch}")

                e_layers = maes_encoders[v]["layers"]
                e_heads = maes_encoders[v]["heads"]
                e_embedding = maes_encoders[v]["embedding"]
                e_mlp = maes_encoders[v]["mlp"]

                d_layers = maes_decoders["B"]["layers"]
                d_heads = maes_decoders["B"]["heads"]
                d_embedding = maes_decoders["B"]["embedding"]
                d_mlp = maes_decoders["B"]["mlp"]

                eparams = profile.encoder_transformer_params(
                    layers=e_layers,
                    embed_dim=e_embedding,
                    mlp_dim=e_mlp
                )
                dparams = profile.decoder_transformer_params(
                    layers=d_layers,
                    embed_dim=d_embedding,
                    mlp_dim=d_mlp
                )
                params = eparams + dparams

                eflops = profile.encoder_transformer_flops(
                    num_tokens=num_encoder_tokens,
                    layers=e_layers,
                    embed_dim=e_embedding,
                    heads=e_heads,
                    mlp_dim=e_mlp
                )
                eflops_per_token = e_layers * profile.encoder_flops(1, e_embedding, e_heads, e_mlp)

                dflops = profile.decoder_transformer_flops(
                    num_tokens=num_decoder_tokens,
                    layers=d_layers,
                    embed_dim=d_embedding,
                    heads=d_heads,
                    mlp_dim=d_mlp
                )
                dflops_per_token = d_layers * profile.decoder_flops(1, d_embedding, d_heads, d_mlp)

                flops = eflops + dflops
                flops_per_patch = eflops_per_token + dflops_per_token

                model_inference_memory = profile.transformer_inference_memory_footprint(
                    params=params,
                    dtype=dtype
                )

                model_training_memory = profile.transformer_training_memory_footprint(
                    params=params,
                    dtype=dtype
                )

                inference_time_per_volume = profile.compute_time(flops=flops, gpu="H100", unit="seconds")
                training_time_per_volume = profile.compute_time(flops=3 * flops, gpu="H100", unit="seconds")
                gflops = np.round(flops / 1e9, 3)
                gflops_per_patch = np.round(flops_per_patch / 1e9, 3)

                patches_per_volume = np.product([s // p for s, p in zip(volume_size, patch_size)])
                pixels_per_patch = np.product(patch_size)
                volumes_per_h100 = 80 // memory_per_volume

                mae_configs[f"{dims} MAE {v}/{patch}"] = {
                    "data": dims,
                    "class": f"{v}/{patch}",
                    "transformer": "autoencoder",
                    "encoder_layers": e_layers,
                    "encoder_heads": e_heads,
                    "encoder_mlp": e_mlp,
                    "encoder_embedding": e_embedding,
                    "decoder_layers": d_layers,
                    "decoder_heads": d_heads,
                    "decoder_mlp": d_mlp,
                    "decoder_embedding": d_embedding,
                    "t": maes_dimensions[dims]["t"],
                    "x": maes_dimensions[dims]["x"],
                    "y": maes_dimensions[dims]["y"],
                    "z": maes_dimensions[dims]["z"],
                    "c": maes_dimensions[dims]["c"],
                    "pt": patches[dims]["t"],
                    "px": patches[dims]["x"],
                    "py": patches[dims]["y"],
                    "pz": patches[dims]["z"],
                    "pc": patches[dims]["c"],
                    "patches_per_volume": patches_per_volume,
                    "pixels_per_patch": pixels_per_patch,
                    "volumes_per_h100": volumes_per_h100,
                    "memory_per_volume": memory_per_volume,
                    "parameters": params,
                    "inference_gflops_per_volume": gflops,
                    "training_gflops_per_volume": 3 * gflops,
                    "gflops_per_patch": gflops_per_patch,
                    "model_inference_memory": model_inference_memory,
                    "model_training_memory": model_training_memory,
                    "inference_time_per_volume": inference_time_per_volume,
                    "training_time_per_volume": training_time_per_volume,
                }

    mae_scaling = pd.DataFrame.from_dict(mae_configs, orient='index')
    mae_scaling = mae_scaling.sort_values(['px', 'parameters', 'encoder_layers', 'encoder_heads'], ascending=[True, True, True, True])
    mae_scaling.to_csv(outdir / "maes_ssl.csv")
    return mae_scaling


def scaling_mae_ft(
    ishape={'t': 16, 'z': 128, 'y': 128, 'x': 128, 'c': 3},
    dtype='float16',  # per channel
    outdir=Path("../scaling/data/maes/ft"),
    mask_ratio=0.75
):
    maes_dimensions = {
        "2D(g)": {"t": 1, "z": 1, "y": ishape['y'], "x": ishape['x'], "c": 1},
        "2D(rgb)": {"t": 1, "z": 1, "y": ishape['y'], "x": ishape['x'], "c": ishape['c']},
        "3D(g)": {"t": 1, "z": ishape['z'], "y": ishape['y'], "x": ishape['x'], "c": 1},
        "3D(rgb)": {"t": 1, "z": ishape['z'], "y": ishape['y'], "x": ishape['x'], "c": ishape['c']},
        "4D(g)": {"t": ishape['t'], "z": ishape['z'], "y": ishape['y'], "x": ishape['x'], "c": 1},
        "4D(rgb)": {"t": ishape['t'], "z": ishape['z'], "y": ishape['y'], "x": ishape['x'], "c": ishape['c']}
    }
    maes_encoders = {
        "S": {"layers": 12, "heads": 6, "embedding": 384, "mlp": 1536},
        "B": {"layers": 12, "heads": 12, "embedding": 768, "mlp": 3072},
        "L": {"layers": 24, "heads": 16, "embedding": 1024, "mlp": 4096},
        "H": {"layers": 32, "heads": 16, "embedding": 1280, "mlp": 5120},
        "2B": {"layers": 24, "heads": 32, "embedding": 2560, "mlp": 10240},
        "6B": {"layers": 32, "heads": 32, "embedding": 4096, "mlp": 16384},
        "g": {"layers": 40, "heads": 16, "embedding": 1408, "mlp": 6144},
        "G": {"layers": 48, "heads": 16, "embedding": 1664, "mlp": 8192},
        "e": {"layers": 56, "heads": 16, "embedding": 1792, "mlp": 15360},
        "22B": {"layers": 48, "heads": 48, "embedding": 6144, "mlp": 24576},
    }
    maes_decoders = {
        "B": {"layers": 4, "heads": 16, "embedding": 512, "mlp": 2048},
    }

    mae_configs = {}
    for patch in [14, 16]:
        patches = {
            "2D(g)": {"t": 1, "z": 1, "y": patch, "x": patch, "c": 1},
            "2D(rgb)": {"t": 1, "z": 1, "y": patch, "x": patch, "c": 3},
            "3D(g)": {"t": 1, "z": patch, "y": patch, "x": patch, "c": 1},
            "3D(rgb)": {"t": 1, "z": patch, "y": patch, "x": patch, "c": 3},
            "4D(g)": {"t": 2, "z": patch, "y": patch, "x": patch, "c": 1},
            "4D(rgb)": {"t": 2, "z": patch, "y": patch, "x": patch, "c": 3}
        }

        for dims in maes_dimensions.keys():

            volume_size = list(maes_dimensions[dims].values())
            patch_size = list(patches[dims].values())
            num_encoder_tokens = profile.patchify(volume_size=volume_size, patch_size=patch_size, mask_ratio=mask_ratio)
            num_decoder_tokens = profile.patchify(volume_size=volume_size, patch_size=patch_size, mask_ratio=0.0)

            memory_per_volume = profile.data_memory_footprint(
                volume_size=volume_size,
                batch_size=1,
                dtype=dtype,
            )

            for v in maes_encoders:
                print(f"{dims} MAE {v}/{patch}")

                e_layers = maes_encoders[v]["layers"]
                e_heads = maes_encoders[v]["heads"]
                e_embedding = maes_encoders[v]["embedding"]
                e_mlp = maes_encoders[v]["mlp"]

                d_layers = maes_decoders["B"]["layers"]
                d_heads = maes_decoders["B"]["heads"]
                d_embedding = maes_decoders["B"]["embedding"]
                d_mlp = maes_decoders["B"]["mlp"]

                eparams = profile.encoder_transformer_params(
                    layers=e_layers,
                    embed_dim=e_embedding,
                    mlp_dim=e_mlp
                )
                dparams = profile.decoder_transformer_params(
                    layers=d_layers,
                    embed_dim=d_embedding,
                    mlp_dim=d_mlp
                )

                eflops = profile.encoder_transformer_flops(
                    num_tokens=num_encoder_tokens,
                    layers=e_layers,
                    embed_dim=e_embedding,
                    heads=e_heads,
                    mlp_dim=e_mlp
                )
                eflops_per_token = e_layers * profile.encoder_flops(1, e_embedding, e_heads, e_mlp)

                dflops = profile.decoder_transformer_flops(
                    num_tokens=num_decoder_tokens,
                    layers=d_layers,
                    embed_dim=d_embedding,
                    heads=d_heads,
                    mlp_dim=d_mlp
                )
                dflops_per_token = d_layers * profile.decoder_flops(1, d_embedding, d_heads, d_mlp)

                params = eparams + dparams

                encoder_inference_memory = profile.transformer_inference_memory_footprint(
                    params=eparams,
                    dtype=dtype
                )

                decoder_inference_memory = profile.transformer_inference_memory_footprint(
                    params=dparams,
                    dtype=dtype
                )

                decoder_training_memory = profile.transformer_training_memory_footprint(
                    params=dparams,
                    dtype=dtype
                )

                model_inference_memory = encoder_inference_memory + decoder_inference_memory

                # encoder weights are frozen during training
                model_training_memory = encoder_inference_memory + decoder_training_memory

                encoder_inference_time_per_volume = profile.compute_time(
                    flops=eflops,
                    gpu="H100",
                    unit="seconds"
                )

                decoder_inference_time_per_volume = profile.compute_time(
                    flops=dflops,
                    gpu="H100",
                    unit="seconds"
                )

                inference_time_per_volume = encoder_inference_time_per_volume + decoder_inference_time_per_volume

                decoder_training_time_per_volume = profile.compute_time(
                    flops=3 * dflops,
                    gpu="H100",
                    unit="seconds"
                )

                # encoder weights are frozen during training
                training_time_per_volume = encoder_inference_time_per_volume + decoder_training_time_per_volume

                inference_flops = eflops + dflops
                training_flops = eflops + (3 * dflops) # encoder weights are frozen during training

                inference_gflops = np.round(inference_flops / 1e9, 3)
                training_gflops = np.round(training_flops / 1e9, 3)

                flops_per_patch = eflops_per_token + dflops_per_token
                gflops_per_patch = np.round(flops_per_patch / 1e9, 3)

                patches_per_volume = np.product([s // p for s, p in zip(volume_size, patch_size)])
                pixels_per_patch = np.product(patch_size)
                volumes_per_h100 = 80 // memory_per_volume

                mae_configs[f"{dims} MAE {v}/{patch}"] = {
                    "data": dims,
                    "class": f"{v}/{patch}",
                    "transformer": "autoencoder",
                    "encoder_layers": e_layers,
                    "encoder_heads": e_heads,
                    "encoder_mlp": e_mlp,
                    "encoder_embedding": e_embedding,
                    "decoder_layers": d_layers,
                    "decoder_heads": d_heads,
                    "decoder_mlp": d_mlp,
                    "decoder_embedding": d_embedding,
                    "t": maes_dimensions[dims]["t"],
                    "x": maes_dimensions[dims]["x"],
                    "y": maes_dimensions[dims]["y"],
                    "z": maes_dimensions[dims]["z"],
                    "c": maes_dimensions[dims]["c"],
                    "pt": patches[dims]["t"],
                    "px": patches[dims]["x"],
                    "py": patches[dims]["y"],
                    "pz": patches[dims]["z"],
                    "pc": patches[dims]["c"],
                    "patches_per_volume": patches_per_volume,
                    "pixels_per_patch": pixels_per_patch,
                    "volumes_per_h100": volumes_per_h100,
                    "memory_per_volume": memory_per_volume,
                    "parameters": params,
                    "inference_gflops_per_volume": inference_gflops,
                    "training_gflops_per_volume": training_gflops,
                    "gflops_per_patch": gflops_per_patch,
                    "model_inference_memory": model_inference_memory,
                    "model_training_memory": model_training_memory,
                    "inference_time_per_volume": inference_time_per_volume,
                    "training_time_per_volume": training_time_per_volume,
                }

    mae_scaling = pd.DataFrame.from_dict(mae_configs, orient='index')
    mae_scaling = mae_scaling.sort_values(['px', 'parameters', 'encoder_layers', 'encoder_heads'], ascending=[True, True, True, True])
    mae_scaling.to_csv(outdir / "maes_ft.csv")
    return mae_scaling



def main(args=None):

    timeit = time.time()
    args = parse_args(args)
    logger.info(args)

    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.arch == "published_models":
        models = {
            "ViT":{
                "S": {
                    "ishape": {'t': 1, 'z': 1, 'y': 224, 'x': 224, 'c': 3},
                    "dataset": "ImageNet-21K",
                    "dataset_size": 14197122,
                    "epochs": 7,
                    "steps": 14197122 * 7 / 4096,
                    "batch_size": 4096
                },
                "B": {
                    "ishape": {'t': 1, 'z': 1, 'y': 224, 'x': 224, 'c': 3},
                    "dataset": "ImageNet-21K",
                    "dataset_size": 14197122,
                    "epochs": 7,
                    "steps": 14197122 * 7 / 4096,
                    "batch_size": 4096
                },
                "L": {
                    "ishape": {'t': 1, 'z': 1, 'y': 224, 'x': 224, 'c': 3},
                    "dataset": "JFT-300M",
                    "dataset_size": 303000000,
                    "epochs": 14,
                    "steps": 1000000,
                    "batch_size": 4096
                },
                "H": {
                    "ishape": {'t': 1, 'z': 1, 'y': 224, 'x': 224, 'c': 3},
                    "dataset": "JFT-300M",
                    "dataset_size": 303000000,
                    "epochs": 14,
                    "steps": 1000000,
                    "batch_size": 4096
                },
                "g": {
                    "ishape": {'t': 1, 'z': 1, 'y': 224, 'x': 224, 'c': 3},
                    "dataset": "JFT-1B",
                    "dataset_size": 3000000000,
                    "epochs": 4000000 * 4096 / 3000000000,
                    "steps": 4000000,
                    "batch_size": 4096
                },
                "G": {
                    "ishape": {'t': 1, 'z': 1, 'y': 224, 'x': 224, 'c': 3},
                    "dataset": "JFT-3B",
                    "dataset_size": 3000000000,
                    "epochs": 5000000 * 4096 / 3000000000,
                    "steps": 5000000,
                    "batch_size": 4096
                },
                "e": {
                    "ishape": {'t': 1, 'z': 1, 'y': 224, 'x': 224, 'c': 3},
                    "dataset": "JFT-3B",
                    "dataset_size": 3000000000,
                    "epochs": 1000000 * 16384 / 3000000000,
                    "steps": 1000000,
                    "batch_size": 16384
                },
                "22B": {
                    "ishape": {'t': 1, 'z': 1, 'y': 224, 'x': 224, 'c': 3},
                    "dataset": "JFT-4B",
                    "dataset_size": 4000000000,
                    "epochs": 177000 * 65000 / 4000000000,
                    "steps": 177000,
                    "batch_size": 65000
                },
            },
            "MAE-SSL":{
                "S": {
                    "ishape": {'t': 1, 'z': 1, 'y': 224, 'x': 224, 'c': 3},
                    "dataset": "ImageNet-1K",
                    "dataset_size": 1281167,
                    "epochs": 1600,
                    "steps": 1281167 * 1600 / 4096,
                    "batch_size": 4096,
                    "mask_ratio": 0.75
                },
                "B": {
                    "ishape": {'t': 1, 'z': 1, 'y': 224, 'x': 224, 'c': 3},
                    "dataset": "ImageNet-1K",
                    "dataset_size": 1281167,
                    "epochs": 1600,
                    "steps": 1281167 * 1600 / 4096,
                    "batch_size": 4096,
                    "mask_ratio": 0.75
                },
                "L": {
                    "ishape": {'t': 1, 'z': 1, 'y': 224, 'x': 224, 'c': 3},
                    "dataset": "ImageNet-1K",
                    "dataset_size": 1281167,
                    "epochs": 1600,
                    "steps": 1281167 * 1600 / 4096,
                    "batch_size": 4096,
                    "mask_ratio": 0.75
                },
                "H": {
                    "ishape": {'t': 1, 'z': 1, 'y': 224, 'x': 224, 'c': 3},
                    "dataset": "Instagram-3B",
                    "dataset_size": 3000000000,
                    "epochs": 1,
                    "steps": 3000000000 * 1 / 4096,
                    "batch_size": 4096,
                    "mask_ratio": 0.75
                },
                "2B": {
                    "ishape": {'t': 1, 'z': 1, 'y': 224, 'x': 224, 'c': 3},
                    "dataset": "Instagram-3B",
                    "dataset_size": 3000000000,
                    "epochs": 4,
                    "steps": 3000000000 * 4 / 4096,
                    "batch_size": 4096,
                    "mask_ratio": 0.75
                },
                "6B": {
                    "ishape": {'t': 1, 'z': 1, 'y': 224, 'x': 224, 'c': 3},
                    "dataset": "Instagram-3B",
                    "dataset_size": 3000000000,
                    "epochs": 4,
                    "steps": 3000000000 * 4 / 4096,
                    "batch_size": 4096,
                    "mask_ratio": 0.75
                },
            },
            "MAE-FT": {
                "S": {
                    "ishape": {'t': 1, 'z': 1, 'y': 518, 'x': 518, 'c': 3},
                    "dataset": "ImageNet-1K",
                    "dataset_size": 1281167,
                    "epochs": 50,
                    "steps": 1281167 * 50 / 1024,
                    "batch_size": 1024
                },
                "B": {
                    "ishape": {'t': 1, 'z': 1, 'y': 518, 'x': 518, 'c': 3},
                    "dataset": "ImageNet-1K",
                    "dataset_size": 1281167,
                    "epochs": 50,
                    "steps": 1281167 * 100 / 1024,
                    "batch_size": 1024
                },
                "L": {
                    "ishape": {'t': 1, 'z': 1, 'y': 518, 'x': 518, 'c': 3},
                    "dataset": "ImageNet-1K",
                    "dataset_size": 1281167,
                    "epochs": 50,
                    "steps": 1281167 * 50 / 1024,
                    "batch_size": 1024
                },
                "H": {
                    "ishape": {'t': 1, 'z': 1, 'y': 518, 'x': 518, 'c': 3},
                    "dataset": "ImageNet-1K",
                    "dataset_size": 1281167,
                    "epochs": 50,
                    "steps": 1281167 * 50 / 1024,
                    "batch_size": 1024
                },
                "2B": {
                    "ishape": {'t': 1, 'z': 1, 'y': 518, 'x': 518, 'c': 3},
                    "dataset": "ImageNet-1K",
                    "dataset_size": 1281167,
                    "epochs": 50,
                    "steps": 1281167 * 50 / 1024,
                    "batch_size": 1024
                },
                "6B": {
                    "ishape": {'t': 1, 'z': 1, 'y': 518, 'x': 518, 'c': 3},
                    "dataset": "ImageNet-1K",
                    "dataset_size": 1281167,
                    "epochs": 50,
                    "steps": 1281167 * 50 / 1024,
                    "batch_size": 1024
                },
            }
        }

        for m in models.keys():
            out = args.outdir / m
            out.mkdir(parents=True, exist_ok=True)

            if m == "ViT":
                df = scaling_vit(
                    ishape=models[m]['S']['ishape'],
                    dtype=args.dtype,
                    outdir=out
                )
            elif m == "MAE-SSL":
                df = scaling_mae_ssl(
                    ishape=models[m]['S']['ishape'],
                    dtype=args.dtype,
                    outdir=out,
                    mask_ratio=models[m]['S']['mask_ratio']
                )

            elif m == "MAE-FT":
                df = scaling_mae_ft(
                    ishape=models[m]['S']['ishape'],
                    dtype=args.dtype,
                    outdir=out,
                )

            df = df.loc[df['data'].str.match(r'2D\(rgb\)')]

            for p in [14, 16]:
                for v, d in models[m].items():
                    idx = df.loc[df['class'].str.match(f"{v}/{p}")].index
                    for col, val in d.items():
                        if col == "ishape":
                            df.loc[idx, val.keys()] = val.values()
                        else:
                            if col not in df.columns:
                                df[col] = np.nan
                            df.loc[idx, col] = val

            df.to_csv(out / f"published_models.csv")

            vis.plot_published_models(
                df,
                outdir=out,
                models=models[m],
                cost_h100_per_hr=args.cost_h100_per_hr
            )

    else:
        summary = args.outdir / "summary"
        summary.mkdir(parents=True, exist_ok=True)

        if args.arch == "vit":
            df = scaling_vit(ishape=args.ishape, dtype=args.dtype, outdir=args.outdir)
        elif args.arch == "mae_ssl":
            df = scaling_mae_ssl(ishape=args.ishape, dtype=args.dtype, outdir=args.outdir, mask_ratio=args.mask_ratio)
        elif args.arch == "mae_ft":
            df = scaling_mae_ft(ishape=args.ishape, dtype=args.dtype, outdir=args.outdir, mask_ratio=args.mask_ratio)
        elif args.arch == "transformer":
            df = scaling_transformer(ishape=args.ishape, dtype=args.dtype, outdir=args.outdir)


        df["number_h100_for_batch"] = np.ceil(df["model_training_memory"] + (df["memory_per_volume"] * args.batch_size) / 80)
        df["cost_h100_for_batch"] = df["number_h100_for_batch"] * 37500
        df["training_h100_hours_per_step"] = args.batch_size * df["training_time_per_volume"] / 3600
        df["training_tflops_per_volume"] = df["training_gflops_per_volume"] / 1000

        for epoch in [1, 50, 100, 300, 500, 800, 1000]:
            for dataset_size in [1000000, 1281167, 14197122, 10000000, 100000000, 303000000, 1000000000]:
                e = "" if epoch == 1 else f"{epoch}_"
                df[f"training_h100_days_per_{e}epoch_{dataset_size}"] = dataset_size * df["training_time_per_volume"] / 3600 / 24 * epoch
                df[f"multigpu_training_days_per_{e}epoch_{dataset_size}"] = df[f"training_h100_days_per_epoch_{dataset_size}"] / df["number_h100_for_batch"] * epoch
                df[f"multigpu_256_training_days_per_{e}epoch_{dataset_size}"] = df[f"training_h100_days_per_epoch_{dataset_size}"] / 256 * epoch
                df[f"training_h100_cost_per_{e}epoch_{dataset_size}"] = df[f"training_h100_days_per_epoch_{dataset_size}"] * 24 * args.cost_h100_per_hr * epoch
                df[f"training_tflops_per_{e}epoch_{dataset_size}"] = df[f"training_tflops_per_volume"] * dataset_size * epoch
                df[f"memory_per_{dataset_size}"] = df[f"memory_per_volume"] * dataset_size
                df[f"num_volumes"] = df[f"memory_per_volume"] * dataset_size

        for arch in df['transformer'].unique():
            data = df[df['transformer'] == arch]

            outdir = args.outdir / arch
            outdir.mkdir(parents=True, exist_ok=True)

            outdir_summary = summary / arch
            outdir_summary.mkdir(parents=True, exist_ok=True)

            vis.plot_data_parameter_scaling(
                data,
                outdir=outdir_summary,
                x="parameters",
                xlabel="Model size (non-embedding parameters)",
                y="number_h100_for_batch",
                ylabel=f"Minimum number of H100s needed for a batch ({args.batch_size})",
                ytwin1="cost_h100_for_batch",
                ytwinlabel1=f"Cost of H100s needed for a batch ({args.batch_size}, $37,500 each)",
                published_models_only=False,
                ylog=True,
                patch_size=args.ipatch["x"],
                cost_h100_per_hr=args.cost_h100_per_hr,
                rgb='rgb' if args.rgb else 'g',
                legend=[
                    'Data (x, y, z, t, c)',
                        f'4D ({args.ishape["x"]}, {args.ishape["y"]}, {args.ishape["z"]}, {args.ishape["t"]}, {args.ishape["c"] if args.rgb else 1})',
                        f'3D ({args.ishape["x"]}, {args.ishape["y"]}, {args.ishape["z"]}, 1, {args.ishape["c"] if args.rgb else 1})',
                        f'2D ({args.ishape["x"]}, {args.ishape["y"]}, 1, 1, {3 if args.rgb else 1})',
                    'Patch (x, y, z, t, c)',
                        f'({args.ipatch["x"]}, {args.ipatch["y"]}, {args.ipatch["z"]}, {args.ipatch["t"]}, {args.ipatch["c"] if args.rgb else 1})',
                ]
            )

            for var in ['days', 'cost']:
                for epoch in [1, 100, 300, 500]:
                    e = "" if epoch == 1 else f"{epoch}_"
                    vis.plot_data_parameter_scaling(
                        data,
                        outdir=outdir_summary,
                        x="parameters",
                        xlabel="Model size (non-embedding parameters)",
                        y=f"training_h100_{var}_per_{e}epoch_1000000",
                        ylabel=f"Training H100 {var} per epoch" if epoch == 1 else f"Training H100 {var} for ({epoch}) epoch(s)",
                        yscalelabel="1M",
                        ytwin1=f"training_h100_{var}_per_{e}epoch_10000000",
                        ytwinlabel1=f"10M",
                        ytwin2=f"training_h100_{var}_per_{e}epoch_100000000",
                        ytwinlabel2=f"100M",
                        ytwin3=f"training_h100_{var}_per_{e}epoch_100000000",
                        ytwinlabel3=f"1B",
                        published_models_only=False,
                        ylog=True,
                        patch_size=args.ipatch["x"],
                        rgb='rgb' if args.rgb else 'g',
                        legend=[
                            'Data (x, y, z, t, c)',
                                f'4D ({args.ishape["x"]}, {args.ishape["y"]}, {args.ishape["z"]}, {args.ishape["t"]}, {args.ishape["c"] if args.rgb else 1})',
                                f'3D ({args.ishape["x"]}, {args.ishape["y"]}, {args.ishape["z"]}, 1, {args.ishape["c"] if args.rgb else 1})',
                                f'2D ({args.ishape["x"]}, {args.ishape["y"]}, 1, 1, {3 if args.rgb else 1})',
                            'Patch (x, y, z, t, c)',
                                f'({args.ipatch["x"]}, {args.ipatch["y"]}, {args.ipatch["z"]}, {args.ipatch["t"]}, {args.ipatch["c"] if args.rgb else 1})',
                        ]
                    )

            vis.plot_individual_parameters(
                data,
                batch_size=args.batch_size,
                outdir=outdir,
                cost_h100_per_hr=args.cost_h100_per_hr,
                rgb='rgb' if args.rgb else 'g',
                legend=[
                    'Data (x, y, z, t, c)',
                    f'4D ({args.ishape["x"]}, {args.ishape["y"]}, {args.ishape["z"]}, {args.ishape["t"]}, {args.ishape["c"] if args.rgb else 1})',
                    f'3D ({args.ishape["x"]}, {args.ishape["y"]}, {args.ishape["z"]}, 1, {args.ishape["c"] if args.rgb else 1})',
                    f'2D ({args.ishape["x"]}, {args.ishape["y"]}, 1, 1, {3 if args.rgb else 1})',
                    'Patch (x, y, z, t, c)',
                    f'({args.ipatch["x"]}, {args.ipatch["y"]}, {args.ipatch["z"]}, {args.ipatch["t"]}, {args.ipatch["c"] if args.rgb else 1})',
                ]
            )

    logger.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
