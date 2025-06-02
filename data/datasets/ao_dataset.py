import logging
import os
import sys
from functools import partial
from multiprocessing import Manager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import ujson
from line_profiler_pycharm import profile
from ray.data import Dataset
from tifffile import TiffFile
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from omegaconf import DictConfig

from utils.preprocessing import resize_with_crop_or_pad
from utils.common import multiprocess
from utils.fourier_embeddings import fourier_embeddings
from data.data_types import TORCH_DTYPES

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@profile
def get_image(path):
    if isinstance(path, torch.Tensor):
        path = Path(str(path.numpy(), "utf-8"))
    else:
        path = Path(str(path))
    
    if path.suffix == '.npy':
        with np.load(path) as arr:
            img = arr
    
    elif path.suffix == '.npz':
        with np.load(path) as data:
            img = data['arr_0']
    
    elif path.suffix == '.tif':
        with TiffFile(path) as tif:
            img = tif.asarray()
    else:
        raise Exception(f"Unknown file format {path}")
    
    if np.isnan(np.sum(img)):
        logger.error("NaN!")
    
    if img.shape[-1] != 1:  # add a channel dim
        img = np.expand_dims(img, axis=-1)
    
    return img.astype(np.float32)


def get_metadata(path, codename: str):
    try:
        if isinstance(path, torch.Tensor):
            path = Path(str(path.numpy(), "utf-8"))
        else:
            path = Path(str(path))
        
        with open(path.with_suffix('.json')) as f:
            hashtbl = ujson.load(f)
        
        return hashtbl[codename]
    
    except KeyError:
        return None


@profile
def get_sample(
    path,
    input_coverage=1.0,
    embedding_option='spatial_planes',
    iotf=None,
    lls_defocus: bool = False,
    defocus_only: bool = False
):
    if isinstance(path, torch.Tensor):
        path = Path(str(path.numpy(), "utf-8"))
    else:
        path = Path(str(path))
    
    with open(path.with_suffix('.json')) as f:
        hashtbl = ujson.load(f)

    lls_defocus_offset = np.nan_to_num(hashtbl.get('lls_defocus_offset', 0), nan=0)

    if defocus_only:
        zernikes = [lls_defocus_offset]
    else:
        zernikes = hashtbl['zernikes']
        
        if lls_defocus:
            zernikes.append(lls_defocus_offset)

    del hashtbl
    zernikes = np.array(zernikes).astype('float32')

    img = get_image(path).astype('float32')

    if input_coverage != 1.:
        img = resize_with_crop_or_pad(img, crop_shape=[int(s * input_coverage) for s in img.shape])

    if img.shape[0] == img.shape[1] and iotf is not None:
        img = fourier_embeddings(
            img,
            iotf=iotf,
            padsize=None,
            alpha_val='abs',
            phi_val='angle',
            remove_interference=True,
            embedding_option=embedding_option,
        )

    return img, zernikes


@profile
def get_metadata(
    path,
    metadata=False,
    input_coverage=1.0,
    embedding_option='spatial_planes',
    iotf=None,
    lls_defocus: bool = False,
    defocus_only: bool = False
):
    if isinstance(path, torch.Tensor):
        path = Path(str(path.numpy(), "utf-8"))
    else:
        path = Path(str(path))

    with open(path.with_suffix('.json')) as f:
        hashtbl = ujson.load(f)

    npoints = int(hashtbl.get('npoints', 1))
    photons = hashtbl.get('photons', 0)
    counts = hashtbl.get('counts', 0)
    counts_mode = hashtbl.get('counts_mode', 0)
    counts_percentiles = hashtbl.get('counts_percentiles', np.zeros(100))

    lls_defocus_offset = np.nan_to_num(hashtbl.get('lls_defocus_offset', 0), nan=0)
    avg_min_distance = np.nan_to_num(hashtbl.get('avg_min_distance', 0), nan=0)

    if defocus_only:
        zernikes = [lls_defocus_offset]
    else:
        zernikes = hashtbl['zernikes']

        if lls_defocus:
            zernikes.append(lls_defocus_offset)

    zernikes = np.array(zernikes).astype('float32')

    try:
        umRMS = hashtbl['umRMS']
    except KeyError:
        umRMS = np.linalg.norm(hashtbl['zernikes'])

    try:
        p2v = hashtbl['peak2peak']
    except KeyError:
        raise Exception(f"Missing peak2peak in {path}")

    if metadata:
        return zernikes, photons, counts, counts_mode, counts_percentiles, p2v, umRMS, npoints, avg_min_distance, str(
            path)

    else:
        img = get_image(path).astype('float32')

        if input_coverage != 1.:
            img = resize_with_crop_or_pad(img, crop_shape=[int(s * input_coverage) for s in img.shape])

        if img.shape[0] == img.shape[1] and iotf is not None:
            img = fourier_embeddings(
                img,
                iotf=iotf,
                padsize=None,
                alpha_val='abs',
                phi_val='angle',
                remove_interference=True,
                embedding_option=embedding_option,
            )

        return img, zernikes


@profile
def check_sample(path):
    try:
        with open(path.with_suffix('.json')) as f:
            ujson.load(f)
        
        with TiffFile(path) as tif:
            tif.asarray()

        with TiffFile(f"{path.with_suffix('')}_realspace.tif") as tif:
            tif.asarray()

        return 1
    
    except Exception as e:
        logger.warning(f"Corrupted file {path}: {e}")
        return path


@profile
def lookup_sample(path):
    if not path.with_suffix('.json').exists():
        logger.warning(f"Missing json file {path}")
        return path

    if not path.exists():
        logger.warning(f"Missing tif file {path}")
        return path

    return 1

@profile
def check_criteria(
    file,
    distribution='/',
    embedding='',
    modes=-1,
    max_amplitude=1.,
    photons_range=None,
    npoints_range=None,
):
    path = str(file)
    amp = float(str([s for s in file.parts if s.startswith('amp_')][0]).split('-')[-1].replace('p', '.'))
    photons = tuple(map(int, str([s.strip('photons_') for s in file.parts if s.startswith('photons_')][0]).split('-')))
    npoints = int([s.strip('npoints_') for s in file.parts if s.startswith('npoints')][0])
    modes = '' if modes - 1 else str(modes)
    
    if 'iter' not in path \
            and (distribution == '/' or distribution in path) \
            and embedding in path \
            and f"z{modes}" in path \
            and amp <= max_amplitude \
            and ((npoints_range[0] <= npoints <= npoints_range[1]) if npoints_range is not None else True) \
            and (
    (photons_range[0] <= photons[0] and photons[1] <= photons_range[1]) if photons_range is not None else True) \
            and lookup_sample(file) == 1:  # access file system only after everything else has passed.
        return path
    else:
        return None


@profile
def collect_files(
    datadir,
    samplelimit=None,
    distribution='/',
    embedding='',
    modes=-1,
    max_amplitude=1.,
    photons_range=None,
    npoints_range=None,
    filename_pattern: str = r"*[!_gt|!_realspace|!_noisefree|!_predictions_psf|!_corrected_psf|!_reconstructed_psf].tif",
    shuffle=True,
    cpu_workers: int = -1,
):
    if not Path(datadir).exists():
        raise Exception(f"The 'datadir' does not exist: {datadir}")
    s1 = f'Searching for files that meet:'
    s2 = f'npoints_range=({int(npoints_range[0]):,} to {int(npoints_range[1]):,} objects),' if npoints_range is not None else ""
    s3 = f'photons_range=({int(photons_range[0]):,} to {int(photons_range[1]):,} photons),' if photons_range is not None else ""
    s4 = f'{max_amplitude=}, number of {modes=}.'
    s5 = f'In data directory: {Path(datadir).resolve()} which exists={Path(datadir).exists()}'
    logger.info(" ".join([s1, s2, s3, s4]))
    logger.info(s5)
    
    check = partial(
        check_criteria,
        distribution=distribution,
        embedding=embedding,
        modes=modes,
        max_amplitude=max_amplitude,
        photons_range=photons_range,
        npoints_range=npoints_range,
    )
    candidate_files = sorted(Path(datadir).rglob(filename_pattern))
    files = multiprocess(
        func=check,
        jobs=candidate_files,
        cores=cpu_workers,
        desc='Loading dataset hashtable',
        unit=' .tif candidates checked'
    )
    try:
        files = [f for f in files if f is not None]
        logger.info(f'.tif files that meet criteria: {len(files)} files')
    except TypeError:
        raise Exception(f'No files that meet criteria out of {len(candidate_files)} candidate files, '
                        f'{sum(len(files) for _, _, files in os.walk(datadir))} total files, '
                        f'in data directory: {Path(datadir).resolve()} which exists={Path(datadir).exists()}')
    
    if samplelimit is not None:
        files = np.random.choice(files, min(samplelimit, len(files)), replace=False).tolist()
        logger.info(f'.tif files selected ({samplelimit=}): {len(files)} files')
    
    if shuffle:
        np.random.shuffle(files)
    
    return files


@profile
def check_dataset(
    datadir,
    cpu_workers: int = 1,
    filename_pattern: str = r"*[!_gt|!_realspace|!_noisefree|!_predictions_psf|!_corrected_psf|!_reconstructed_psf].tif"
):
    jobs = multiprocess(func=check_sample, jobs=sorted(Path(datadir).rglob(filename_pattern)), cores=cpu_workers)
    corrupted = [j for j in jobs if j != 1]
    corrupted = pd.DataFrame(corrupted, columns=['path'])
    logger.info(f"Corrupted files [{corrupted.index.shape[0]}]")

    logger.info(corrupted)

    try:
        corrupted.to_csv(datadir / 'corrupted.csv', header=False, index=False)
        logger.info(datadir / 'corrupted.csv')
    except PermissionError:
        corrupted.to_csv(Path.cwd() / 'corrupted.csv', header=False, index=False)
        logger.info(Path.cwd() / 'corrupted.csv')

    return corrupted



class DataFinder(Dataset):
    def __init__(
        self,
        datadir,
        distribution='/',
        embedding='',
        modes=-1,
        samplelimit=None,
        max_amplitude=1.,
        input_coverage=1.,
        embedding_option='spatial_planes',
        photons_range=None,
        npoints_range=None,
        iotf=None,
        metadata=False,
        lls_defocus: bool = False,
        defocus_only: bool = False,
        filename_pattern: str = r"*[!_gt|!_realspace|!_noisefree|!_predictions_psf|!_corrected_psf|!_reconstructed_psf].tif",
        cpu_workers: int = -1,
        model_input_shape: tuple = (6, 64, 64, 1),
        dtype=torch.float32
    ):
        super(Dataset, self).__init__()
        
        self.datadir = datadir
        self.distribution = distribution
        self.embedding = embedding
        self.modes = modes
        self.samplelimit = samplelimit
        self.max_amplitude = max_amplitude
        self.input_coverage = input_coverage
        self.embedding_option = embedding_option
        self.photons_range = photons_range
        self.npoints_range = npoints_range
        self.iotf = iotf
        self.metadata = metadata
        self.lls_defocus = lls_defocus
        self.defocus_only = defocus_only
        self.filename_pattern = filename_pattern
        self.cpu_workers = cpu_workers
        self.model_input_shape = model_input_shape
        self.dtype = dtype

        manager = Manager()

        self.files = manager.list(
            collect_files(
                datadir,
                modes=self.modes,
                samplelimit=self.samplelimit,
                embedding=self.embedding,
                distribution=self.distribution,
                max_amplitude=self.max_amplitude,
                photons_range=self.photons_range,
                npoints_range=self.npoints_range,
                filename_pattern=self.filename_pattern,
                cpu_workers=self.cpu_workers
        ))

    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx, debug=False):
        if debug:
            worker_info = torch.utils.data.get_worker_info()
            print(f"{worker_info=}")
            print(f"loading {self.files[idx]} using worker {worker_info.id}")

        path = self.files[idx]

        if self.metadata:
            return get_metadata(
                path=path,
                input_coverage=self.input_coverage,
                iotf=self.iotf,
                embedding_option=self.embedding_option,
                metadata=self.metadata,
                lls_defocus=self.lls_defocus,
                defocus_only=self.defocus_only
            )
        else:
            x, y = get_sample(
                path=path,
                iotf=self.iotf,
                input_coverage=self.input_coverage,
                embedding_option=self.embedding_option,
                lls_defocus=self.lls_defocus,
                defocus_only=self.defocus_only
            )
            return torch.tensor(x, dtype=self.dtype), torch.tensor(y, dtype=self.dtype)


@profile
def collect_dataset(config: DictConfig):
    dataset = DataFinder(
        datadir=Path(config.datasets.datadir),
        distribution=config.datasets.distribution,
        modes=config.datasets.modes,
        samplelimit=config.datasets.samplelimit,
        max_amplitude=config.datasets.max_amplitude,
        input_coverage=config.datasets.input_coverage,
        embedding_option=config.datasets.embedding_option,
        photons_range=config.datasets.photons_range,
        npoints_range=config.datasets.npoints_range,
        iotf=config.datasets.iotf,
        metadata=config.datasets.metadata,
        lls_defocus=config.datasets.lls_defocus,
        defocus_only=config.datasets.defocus_only,
        filename_pattern=config.datasets.filename_pattern,
        dtype=TORCH_DTYPES[config.datasets.dtype].value,
        cpu_workers=config.clusters.cpu_workers,
    )

    if config.datasets.return_dataloader:
        if config.datasets.split is not None:
            val_size = round(len(dataset) * config.datasets.split)
            train, val = random_split(dataset, lengths=[len(dataset) - val_size, val_size])

            train = DataLoader(
                train,
                collate_fn=config.datasets.collate_fn,
                batch_size=config.clusters.worker_batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=config.clusters.gpu_workers,
                prefetch_factor=2,
                persistent_workers=False,
                sampler=DistributedSampler(dataset)
            )
            val = DataLoader(
                val,
                collate_fn=config.datasets.collate_fn,
                batch_size=config.clusters.worker_batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=config.clusters.gpu_workers,
                prefetch_factor=2,
                persistent_workers=False,
                sampler=DistributedSampler(dataset)
            )

            i = next(iter(train))
            logger.info(f"Input: {i[0].shape}")
            logger.info(f"Output: {i[1].shape}")
            logger.info(f"Training batches: {len(train)}")
            logger.info(f"Validation batches: {len(val)}")

            return train, val

        else:

            data = DataLoader(
                dataset,
                collate_fn=config.datasets.collate_fn,
                batch_size=config.clusters.worker_batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=config.clusters.gpu_workers,
                prefetch_factor=2,
                persistent_workers=False,
                sampler=DistributedSampler(dataset)
            )

            try:
                if not config.datasets.metadata:
                    i = next(iter(data))
                    logger.info(f"Input: {i[0].shape}")

                    logger.info(f"Output: {i[1].shape}")
                    logger.info(f"Batches: {len(data)}")
            except Exception as e:
                logger.warning(e)

            return data

    else:
        return dataset

