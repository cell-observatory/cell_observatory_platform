import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use('Agg')

import logging
import sys
import time
from pathlib import Path

import cli
import numpy as np
import pandas as pd
import vis

import utilization

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args(args):
    parser = cli.argparser()
    
    parser.add_argument(
        "--datadir", default="/groups/betzig/betziglab/thayer/pretrained_models/pretraining", type=Path, help='path to pretrained models'
    )

    parser.add_argument(
        "--outdir", default="../utilization/data", type=Path, help='path to save plots'
    )

    parser.add_argument(
        "--order", default='t-z-y-x-c', help='order of dimensions', type=str,
    )

    parser.add_argument(
        "--pretraining_token_shape", default="16-128-256-256-2", type=str, help='image/video shape for pretraining data'
    )

    parser.add_argument(
        "--pretraining_token_channel_dtype", default="fp16", type=str, help='channel dtype for pretraining data'
    )
    
    parser.add_argument(
        "--wandb_project", default="profiling", type=str, help='W&B project name'
    )

    return parser.parse_known_args(args)[0]

def main(args=None):
    timeit = time.time()
    args = parse_args(args)
    logger.info(args)

    args.outdir = args.outdir 
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    args.pretraining_token_shape = {
        o: int(i) for o, i in zip(args.order.split('-'), args.pretraining_token_shape.split('-'))
    } if isinstance(args.pretraining_token_shape, str) else args.pretraining_token_shape

    
    logger.info(f"Getting utilization from {args.datadir}...")
    
    df = utilization.get_utilization(
        datadir=args.datadir,
        outdir=args.outdir,
        wandb_project=args.wandb_project,
        pretraining_token_shape=args.pretraining_token_shape,
        pretraining_token_channel_dtype=args.pretraining_token_channel_dtype,
    )
    vis.plot_gpu_scaling(df, args.outdir)
    vis.plot_model_scaling(df, args.outdir)
    vis.plot_utilization(df, args.outdir)
    vis.plot_flops(df, args.outdir)
    vis.plot_data_scaling(df, args.outdir)
    
    logger.info(f"Saving plots to {args.outdir}...")

    logger.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
