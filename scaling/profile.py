import matplotlib
matplotlib.use('Agg')

import sys
import logging
import numpy as np

try:
    import cupy as cp
except ImportError as e:
    logging.warning(f"Cupy not supported on your system: {e}")

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def matmul_flops(m, n, k):
    return m * (2 * n - 1) * k


def softmax_flops(n, d):
    return n * (3 * d - 1)


def layernorm_flops(n, d):
    flops = n * d - 1  # mean
    flops += 3 * n * d - 1  # variance
    flops += 2 * n * d  # normalize
    return flops


def self_attention_flops(num_tokens, input_dim, output_dim):
    flops = 3 * matmul_flops(num_tokens, input_dim, output_dim)  # Q, K and V
    flops += matmul_flops(num_tokens, output_dim, num_tokens)  # Q*K^T
    flops += num_tokens ** 2
    flops += softmax_flops(num_tokens, num_tokens)
    flops += matmul_flops(num_tokens, num_tokens, output_dim)  # softmax
    return flops


def msa_flops(num_tokens, embed_dim, heads):
    sa_flops = self_attention_flops(
        num_tokens, embed_dim, embed_dim // heads
    )
    flops = heads * sa_flops
    flops += matmul_flops(num_tokens, embed_dim, embed_dim)
    return flops


def mlp_flops(num_tokens, embed_dim, mlp_dim):
    flops = matmul_flops(num_tokens, embed_dim, mlp_dim)  # expand layer
    flops += num_tokens * mlp_dim  # activation
    flops += matmul_flops(num_tokens, mlp_dim, embed_dim)  # project layer
    return flops


def encoder_flops(num_tokens, embed_dim, heads, mlp_dim):
    flops = layernorm_flops(n=num_tokens, d=embed_dim)
    flops += msa_flops(num_tokens, embed_dim, heads)
    flops += num_tokens * embed_dim  # res connection

    flops += layernorm_flops(n=num_tokens, d=embed_dim)
    flops += mlp_flops(num_tokens, embed_dim, mlp_dim)
    flops += num_tokens * embed_dim  # res connection
    return flops


def decoder_flops(num_tokens, embed_dim, heads, mlp_dim):
    flops = layernorm_flops(n=num_tokens, d=embed_dim)
    flops += msa_flops(num_tokens, embed_dim, heads)
    flops += num_tokens * embed_dim  # res connection

    flops += layernorm_flops(n=num_tokens, d=embed_dim)
    flops += msa_flops(num_tokens, embed_dim, heads)  # second msa
    flops += num_tokens * embed_dim  # res connection

    flops += layernorm_flops(n=num_tokens, d=embed_dim)
    flops += mlp_flops(num_tokens, embed_dim, mlp_dim)
    flops += num_tokens * embed_dim  # res connection
    return flops


def patchify_flops(num_tokens, patch_size, embed_dim):
    latent_dim = np.product(patch_size)
    flops = matmul_flops(num_tokens, latent_dim, embed_dim)
    flops += num_tokens * embed_dim  # Add positional embedding
    return flops


def patchify(volume_size, patch_size, class_embedding=False, mask_ratio=0.):
    num_tokens = np.product([s // p for s, p in zip(volume_size, patch_size)])

    if mask_ratio > 0:
        num_tokens = round(num_tokens * (1 - mask_ratio))

    if class_embedding:
        num_tokens += 1

    return num_tokens


def encoder_transformer_flops(num_tokens, layers, embed_dim, heads, mlp_dim):
    flops = layers * encoder_flops(num_tokens, embed_dim, heads, mlp_dim)
    # flops += patchify_flops(num_tokens, patch_size, embed_dim)

    gflops = np.round(flops / 1e9, 3)
    logger.info(f"{flops:,} FLOPs = {gflops} GFLOPs")
    return flops


def decoder_transformer_flops(num_tokens, layers, embed_dim, heads, mlp_dim):
    flops = layers * decoder_flops(num_tokens, embed_dim, heads, mlp_dim)
    # flops += patchify_flops(num_tokens, patch_size, embed_dim)

    gflops = np.round(flops / 1e9, 3)
    logger.info(f"{flops:,} FLOPs = {gflops} GFLOPs")
    return flops


def encoder_transformer_params(layers, embed_dim, mlp_dim):
    attention = 4 * (embed_dim ** 2 + embed_dim)
    feed_forward = 2 * embed_dim * mlp_dim + embed_dim + mlp_dim
    layer_norm = 2 * embed_dim

    encoder = layers * (attention + feed_forward + 2 * layer_norm)
    encoder_mparams = np.round(encoder / 1e6, 0).astype(int)

    logger.info(f"{encoder:,} params = {encoder_mparams} M params")
    # feed_forward = 8 * embed_dim**2 + 5 * embed_dim
    # attention = (4 * embed_dim ** 2 + 4 * embed_dim) * 1
    # layer_norm = (2 * embed_dim) * 2
    # params = 12 * embed_dim**2 + 13 * embed_dim
    return encoder


def decoder_transformer_params(layers, embed_dim, mlp_dim):
    attention = 4 * (embed_dim ** 2 + embed_dim)
    feed_forward = 2 * embed_dim * mlp_dim + embed_dim + mlp_dim
    layer_norm = 2 * embed_dim

    decoder = layers * (2 * attention + feed_forward + 3 * layer_norm)
    decoder_mparams = np.round(decoder / 1e6, 0).astype(int)

    logger.info(f"{decoder:,} params = {decoder_mparams} M params")
    # feed_forward = 8 * embed_dim**2 + 5 * embed_dim
    # attention = 8 * embed_dim ** 2 + 8 * embed_dim
    # layer_norm = 6 * embed_dim
    # params = 16 * embed_dim**2 + 19 * embed_dim
    return decoder


def transformer_inference_memory_footprint(params, dtype='float32'):
    if dtype == 'float16':
        s = 2.0
    elif dtype == 'float32':
        s = 4.0
    elif dtype == 'float64':
        s = 8.0
    else:
        s = 4.0

    mem = int(s * params)
    gbytes = mem / (1024.0 ** 3)

    logger.info(f"{mem:,d} B = {gbytes} GB using ({dtype})")
    return gbytes


def transformer_training_memory_footprint(params, dtype='float32'):
    if dtype == 'float16':
        s = 2.0
    elif dtype == 'float32':
        s = 4.0
    elif dtype == 'float64':
        s = 8.0
    else:
        s = 4.0

    model = s * params
    grads = s * params
    opt = 2 * s * params

    mem = int(model + grads + opt)
    gbytes = mem / (1024.0 ** 3)

    logger.info(f"{mem:,d} B = {gbytes} GB using ({dtype})")
    return gbytes


def data_memory_footprint(volume_size, batch_size=1, dtype='float32'):
    if dtype == 'float8':
        s = 1.0
    elif dtype == 'float16':
        s = 2.0
    elif dtype == 'float32':
        s = 4.0
    elif dtype == 'float64':
        s = 8.0
    else:
        s = 4.0

    mem = int(s * batch_size * np.product(volume_size))
    gbytes = mem / (1024.0 ** 3)

    logger.info(f"{mem:,d} B = {gbytes} GB using ({dtype})")
    return gbytes


def compute_time(flops, gpu="H100", unit="seconds"):
    """
    Google benchmark
    https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/community-content/vertex_model_garden/benchmarking_reports/jax_vit_benchmarking_report.md
    # g/14 533, GFLOPs, 1066 training GFLOPs
    # GFLOPS = GFLOPs / sec/img/GPU
    """
    if gpu == "TPUv3":
        # https://cloud.google.com/tpu/docs/v3
        # 18 images/sec
        # t = 0.055268 sec/img/GPU
        average_utilization = 19 * 10 ** 12

    elif gpu == "A100":
        # https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
        # 51 images/sec
        # t = 0.019608 sec/img/GPU
        average_utilization = 54 * 10 ** 12

    elif gpu == "H100":
        # https://resources.nvidia.com/en-us-tensor-core
        # x2 A100
        average_utilization = 109 * 10 ** 12

    else:
        raise Exception("Unknown GPU device")

    time = flops / average_utilization
    if unit == "hours":
        time = time / (60 * 60)
    elif unit == "minutes":
        time = time / 60
    return time
