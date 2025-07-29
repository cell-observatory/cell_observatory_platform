import matplotlib

matplotlib.use('Agg')

import pickle
import sys, re
import logging
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from astropy import convolution
import multiprocessing as mp
from line_profiler_pycharm import profile
from typing import Any, List, Union, Optional, Generator

try:
    import cupy as cp
except ImportError as e:
    logging.warning(f"Cupy not supported on your system: {e}")

import matplotlib.pyplot as plt

plt.set_loglevel('error')

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@profile
def multiprocess(
        jobs: Union[Generator, List, np.ndarray],
        func: Any,
        desc: str = 'Processing',
        cores: int = -1,
        unit: str = 'it',
        pool: Optional[mp.Pool] = None,
):
    """ Multiprocess a generic function
    Args:
        func: a python function
        jobs: a list of jobs for function `func`
        desc: description for the progress bar
        cores: number of cores to use

    Returns:
        an array of outputs for every function call
    """

    cores = cores if mp.current_process().name == 'MainProcess' else 1
    mp.set_start_method('spawn', force=True)
    jobs = list(jobs)

    if cores == 1 or len(jobs) == 1:
        results = []
        for j in tqdm(
                jobs,
                total=len(jobs),
                desc=desc,
                bar_format='{l_bar}{bar}{r_bar} {elapsed_s:.1f}s elapsed',
                unit=unit,
                file=sys.stdout,
        ):
            results.append(func(j))
    elif cores == -1 and len(jobs) > 0:
        with pool if pool is not None else mp.Pool(min(mp.cpu_count(), len(jobs))) as p:
            results = list(tqdm(
                p.imap(func, jobs),
                total=len(jobs),
                desc=desc,
                bar_format='{l_bar}{bar}{r_bar} {elapsed_s:.1f}s elapsed',
                unit=unit,
                file=sys.stdout,
            ))
    elif cores > 1 and len(jobs) > 0:
        with pool if pool is not None else mp.Pool(cores) as p:
            results = list(tqdm(
                p.imap(func, jobs),
                total=len(jobs),
                desc=desc,
                bar_format='{l_bar}{bar}{r_bar} {elapsed_s:.1f}s elapsed',
                unit=unit,
                file=sys.stdout,
            ))
    else:
        raise Exception(f'No data found in {jobs=}')

    return np.array(results)


def photons2electrons(image, quantum_efficiency: float = .82):
    return image * quantum_efficiency


def electrons2photons(image, quantum_efficiency: float = .82):
    return image / quantum_efficiency


def electrons2counts(image, electrons_per_count: float = .22):
    return image / electrons_per_count


def counts2electrons(image, electrons_per_count: float = .22):
    return image * electrons_per_count


def randuniform(var):
    """
    Returns a random number (uniform chance) in the range provided by var. If var is a scalar, var is simply returned.

    Args:
        var : (as scalar) Returned as is.
        var : (as list) Range to provide a random number

    Returns:
        _type_: ndarray or scalar. Random sample from the range provided.

    """
    var = (var, var) if np.isscalar(var) else var

    # star unpacks a list, so that var's values become the separate arguments here
    return np.random.uniform(*var)


def normal_noise(mean: float, sigma: float, size: tuple) -> np.array:
    mean = randuniform(mean)
    sigma = randuniform(sigma)
    return np.random.normal(loc=mean, scale=sigma, size=size).astype(np.float32)


def poisson_noise(image: np.ndarray) -> np.array:
    image = np.nan_to_num(image, nan=0)
    return np.random.poisson(lam=image).astype(np.float32) - image


def add_noise(
        image: np.ndarray,
        mean_background_offset: int = 100,
        sigma_background_noise: int = 40,
        quantum_efficiency: float = .82,
        electrons_per_count: float = .22,
):
    """

    Args:
        image: noise-free image in incident photons
        mean_background_offset: camera background offset
        sigma_background_noise: read noise from the camera
        quantum_efficiency: quantum efficiency of the camera
        electrons_per_count: conversion factor to go from electrons to counts

    Returns:
        noisy image in counts
    """
    image = photons2electrons(image, quantum_efficiency=quantum_efficiency)
    sigma_background_noise *= electrons_per_count  # electrons;  40 counts = 40 * .22 electrons per count
    dark_read_noise = normal_noise(mean=0, sigma=sigma_background_noise, size=image.shape)  # dark image in electrons
    shot_noise = poisson_noise(image)  # shot noise in electrons

    image += shot_noise + dark_read_noise
    image = electrons2counts(image, electrons_per_count=electrons_per_count)

    image += mean_background_offset  # add camera offset (camera offset in counts)
    image[image < 0] = 0
    return image.astype(np.float32)


def microns2waves(a, wavelength):
    return a / wavelength


def waves2microns(a, wavelength):
    return a * wavelength


def mae(y: np.array, p: np.array, axis=0) -> np.array:
    error = np.abs(y - p)
    return np.mean(error[np.isfinite(error)], axis=axis)


def mse(y: np.array, p: np.array, axis=0) -> np.array:
    error = (y - p) ** 2
    return np.mean(error[np.isfinite(error)], axis=axis)


def rmse(y: np.array, p: np.array, axis=0) -> np.array:
    error = np.sqrt((y - p) ** 2)
    return np.mean(error[np.isfinite(error)], axis=axis)


def mape(y: np.array, p: np.array, axis=0) -> np.array:
    error = np.abs(y - p) / np.abs(y)
    return 100 * np.mean(error[np.isfinite(error)], axis=axis)


@profile
def fftconvolution(kernel, sample):
    if kernel.shape[0] == 1 or kernel.shape[-1] == 1:
        kernel = np.squeeze(kernel)

    if sample.shape[0] == 1 or sample.shape[-1] == 1:
        sample = np.squeeze(sample)

    conv = convolution.convolve_fft(
        sample,
        kernel,
        allow_huge=True,
        normalize_kernel=False,
        nan_treatment='fill',
        fill_value=0
    ).astype(sample.dtype)  # otherwise returns as float64
    conv[conv < 0] = 0  # clip negative small values
    return conv


def fft_decon(kernel, sample, iters):
    for k in range(kernel.ndim):
        kernel = np.roll(kernel, kernel.shape[k] // 2, axis=k)

    kernel = cp.array(kernel)
    sample = cp.array(sample)
    deconv = cp.array(sample)

    kernel = cp.fft.rfftn(kernel)

    for _ in range(iters):
        conv = cp.fft.irfftn(cp.fft.rfftn(deconv) * kernel)
        relative_blur = sample / conv
        deconv *= cp.fft.irfftn((cp.fft.rfftn(relative_blur).conj() * kernel).conj())

    return cp.asnumpy(deconv)


@profile
def percentile_filter(data: np.ndarray, min_pct: int = 5, max_pct: int = 95) -> np.ndarray:
    minval, maxval = np.percentile(data, [min_pct, max_pct])
    return (data < minval) | (data > maxval)


def convert_to_windows_file_string(f):
    f = str(f).replace('/', '\\').replace("\\clusterfs\\nvme\\", "V:\\")
    f = f.replace("\\clusterfs\\nvme2\\", "U:\\")
    return f


def convert_path_to_other_cam(src_path: Path, dst='B'):
    """
    Returns the Path that corresponds to the file with the other camera.

    Args:
        src_path: Path to existing file
        dst: camera letter (e.g. "B" for CamB, "A" for CamA)

    Returns:
        Path to existing file.

    """
    filename = src_path.name  # 'before_Iter_0000_CamA_ch0_CAM1_stack0000_488nm_0000000msec_0177969396msecAbs_-01x_-01y_-01z_0000t.tif'
    prefix = re.findall(r"^.*Cam", filename)[0] + dst  # 'after_threestacks_Iter_0000_Cam' + 'B'
    ending = re.findall(r"msec_.*", filename)[0]  # 'msec_0160433718msecAbs_-01x_-01y_-01z_0000t.tif'
    cam_b = src_path.parent.glob(f"{prefix}*{ending}")
    return list(cam_b)[0]


def round_to_even(n):
    answer = round(n)
    if not answer % 2:
        return int(answer)
    if abs(answer + 1 - n) < abs(answer - 1 - n):
        return int(answer + 1)
    else:
        return int(answer - 1)


def round_to_odd(n):
    answer = round(n)
    if answer % 2:
        return int(answer)
    if abs(answer + 1 - n) < abs(answer - 1 - n):
        return int(answer + 1)
    else:
        return int(answer - 1)


def gaussian_kernel(kernlen: tuple = (21, 21, 21), std=3):
    """Returns a 3D Gaussian kernel array."""
    x = np.arange((-kernlen[2] // 2) + 1, (-kernlen[2] // 2) + 1 + kernlen[2], 1)
    y = np.arange((-kernlen[1] // 2) + 1, (-kernlen[1] // 2) + 1 + kernlen[1], 1)
    z = np.arange((-kernlen[0] // 2) + 1, (-kernlen[0] // 2) + 1 + kernlen[0], 1)
    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
    kernel = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * std ** 2))
    return kernel / np.nansum(kernel)


def fwhm2sigma(w):
    """ convert from full width at half maximum (FWHM) to std """
    return w / (2 * np.sqrt(2 * np.log(2)))


def sigma2fwhm(s):
    """ convert from std to full width at half maximum (FWHM) """
    return s * (2 * np.sqrt(2 * np.log(2)))


def sphere_mask(image_shape, radius=1):
    """
    Args:
        image_shape:
        radius:

    Returns:
        3D Boolean array where True within the sphere
    """
    center = [s // 2 for s in image_shape]
    Z, Y, X = np.ogrid[:image_shape[0], :image_shape[1], :image_shape[2]]
    dist_from_center = np.sqrt((Z - center[0]) ** 2 + (Y - center[1]) ** 2 + (X - center[2]) ** 2)
    mask = dist_from_center <= radius
    return mask


@profile
def fft(inputs, padsize=None):
    if padsize is not None:
        shape = inputs.shape[1]
        size = shape * (padsize / shape)
        pad = int((size - shape) // 2)
        inputs = np.pad(inputs, ((pad, pad), (pad, pad), (pad, pad)), 'constant', constant_values=0)

    otf = np.fft.ifftshift(inputs)
    otf = np.fft.fftn(otf)
    otf = np.fft.fftshift(otf)
    return otf


@profile
def ifft(otf):
    psf = np.fft.ifftshift(otf)
    psf = np.fft.ifftn(psf)
    psf = np.fft.fftshift(psf)
    return np.abs(psf)


@profile
def normalize_otf(otf, freq_strength_threshold: float = 0., percentile: bool = False):

    if percentile:
        otf /= np.nanpercentile(np.abs(otf), 99.99)
    else:
        roi = np.abs(otf[sphere_mask(image_shape=otf.shape, radius=3)])
        dc = np.max(roi)
        otf /= np.mean(roi[roi != dc])

    # since the DC has no bearing on aberration: clamp to -1, +1
    otf[otf > 1] = 1
    otf[otf < -1] = -1

    if freq_strength_threshold != 0.:
        otf[np.abs(otf) < freq_strength_threshold] = 0.

    otf = np.nan_to_num(otf, nan=0, neginf=0, posinf=0)
    return otf


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(x, path):
    with open(path, 'wb') as f:
        pickle.dump(x, f, protocol=pickle.HIGHEST_PROTOCOL)


def savesvg(
    fig: plt.Figure,
    savepath: Union[Path, str],
    top: float = 0.9,
    bottom: float = 0.1,
    left: float = 0.1,
    right: float = 0.9,
    hspace: float = 0.35,
    wspace: float = 0.1
):

    plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)
    plt.savefig(savepath, bbox_inches='tight', dpi=300, pad_inches=.25)

    if Path(savepath).suffix == '.svg':
        # Read in the file
        with open(savepath, 'r', encoding="utf-8") as f:
            filedata = f.read()

        # Replace the target string
        filedata = re.sub('height="[0-9]+(\.[0-9]+)pt"', '', filedata)
        filedata = re.sub('width="[0-9]+(\.[0-9]+)pt"', '', filedata)

        # Write the file out again
        with open(savepath, 'w', encoding="utf-8") as f:
            f.write(filedata)
