import os
import re
import logging
import functools
from pathlib import Path
from typing import Callable, Iterable, List, Literal, Optional, Tuple, Any, Dict

import ray
import torch
# from torch import distributed as dist
from ray.actor import ActorHandle

import numpy as np
# import pandas as pd
# import connectorx as cx

from omegaconf import OmegaConf
# from dotenv import load_dotenv

# from cell_observatory_platform.utils.common import ceil_div
from cell_observatory_platform.training.helpers import get_patch_sizes
from cell_observatory_platform.inference.amg import postprocess_sam_preds
from cell_observatory_platform.utils.context import barrier, get_world_size, process_rank
from cell_observatory_platform.models.layers.patch_embeddings import PatchEmbedding, calc_num_patches
from cell_observatory_platform.data.datasets.buffers import slot_info_to_view, BufferManager
# from cell_observatory_platform.inference.utils import (
#     stable_key_owner, 
#     tile_hash, 
# )
from cell_observatory_platform.inference.saver import SaveWorker
from cell_observatory_platform.inference.visualizer import VizWorker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class InferencerWorker:
    def __init__(
        self,
        aggregate_mode: Literal["stitch_volume", "save_local"],
        inference_mode: Literal["sliding_window", "tile", "hypercube"],
        task: Literal["detection", "instance_segmentation", "semantic_segmentation", "upsample_space", "channel_split"],
        outputs_metadata: dict,
        viz_handlers_configs: Dict[str, Dict[str, Any]],
        model: torch.nn.Module,
        # database: pd.DataFrame,
        # use_cached_hypercubes_dataframe: bool,
        # hypercubes_dataframe_path: Path,
        # timepoint_list: Optional[List[int]],
        # z_step_pdf: int,
        input_format: str,
        input_shape: List[int],
        patch_shape: List[Optional[int]],
        decoder_head_type: str,
        roi_tile_list: List[Tuple[int, str]],
        save_dir: Path | str,
        buffer_manager: BufferManager,
        save_mode: Literal["overwrite", "append", "new_image"],
        save_outputs: bool,
        block_on_save: bool,
        save_worker: ActorHandle[SaveWorker],
        vizualize_outputs: bool,
        block_on_viz: bool,
        viz_worker: ActorHandle[VizWorker],
        viz_sampling_policy: Optional[Dict[str, Any]] = None,
        # dtype: torch.dtype = torch.float16,
        # protocol: Literal["binary", "csv", "cursor"] = "binary",
        # dotenv_path: Optional[Path] = Path(__file__).parent.parent.parent / ".env",
        # dbname: Literal["staging", "prod"] = "prod",
        # server_folder_path: Optional[Path] = None,
        # verbose: bool = False,
        # max_hypercubes: Optional[int] = None,
        # max_partitions: int = 10,
        # save_format: Literal["tiff", "zarr"] = "tiff",
        # zarr_chunk_shape: Optional[Tuple[int, ...]] = None,
        # zarr_shard_shape: Optional[Tuple[int, ...]] = None,
        auxiliary_outputs: Optional[Any] = None,
        # pmin: float = 1.0,
        # pmax: float = 99.0,
        # feature_viz_type: Optional[str] = None,
        # pred_boxes_format: Literal["xyzxyz", "cxcyczwhd"] = "xyzxyz",
        # gt_boxes_format: Literal["xyzxyz", "cxcyczwhd"] = "cxcyczwhd",
        # scale_gt_boxes: bool = True,
    ):
        # self.database = database
        # self.hypercubes_dataframe_path = Path(hypercubes_dataframe_path)
        # self.use_cached_hypercubes_dataframe = use_cached_hypercubes_dataframe
        # # TODO: consider alternative methods for only performing inference
        # #       on a subset of timepoints without storing a prediction for all timepoints
        # self.timepoint_list = timepoint_list

        # self.dtype = dtype
        self.input_shape = input_shape
        self.patch_shape = patch_shape
        self.input_format = input_format

        temporal_patch_size, self.axial_patch_size, self.lateral_patch_size = get_patch_sizes(
            input_format=self.input_format, patch_shape=list(*patch_shape)
        )
        _, token_shape = calc_num_patches(
            input_fmt=self.input_format,
            input_shape=self.input_shape,
            patch_shape=tuple(patch_shape),
        )
        self.pe_unpatchify = functools.partial(
            PatchEmbedding.unpatchify,
            temporal_patch_size=temporal_patch_size,
            axial_patch_size=self.axial_patch_size,
            lateral_patch_size=self.lateral_patch_size,
            token_shape=token_shape,
            input_format=self.input_format,
        )
        # NOTE: if input format does not contain 'T' we set patch size to 1
        #       since buffers assume that the T-axis exists
        self.temporal_patch_size = temporal_patch_size if temporal_patch_size else 1
        self.token_shape = self._get_token_shape(token_shape, self.input_format)

        # roi_tile_list is a list of (roi_id, tile_name) tuples
        # to restrict inference to
        self.roi_tile_list = roi_tile_list
        self.roi_list = list(set([x[0] for x in roi_tile_list]))
        self.tile_list = list(set([x[1] for x in roi_tile_list]))

        self.model = model

        self.inference_mode = inference_mode
        self.task = task
        self.aggregate_mode = aggregate_mode
        if self.aggregate_mode == "save_local":
            assert self.task in {"detection", "instance_segmentation", "semantic_segmentation"}, \
                "save_local aggregate_mode only supported for detection and segmentation tasks"

        self.stitch_volume = (self.aggregate_mode == "stitch_volume")

        # self.pmin = pmin
        # self.pmax = pmax
        # self.z_step_pdf = z_step_pdf
        self.vizualize_outputs = vizualize_outputs
        self.save_outputs = save_outputs
        self.save_mode = save_mode
        self.block_on_save = block_on_save
        self.block_on_viz = block_on_viz
        self.save_worker = save_worker
        self.viz_worker = viz_worker
        self.viz_sampling_policy = viz_sampling_policy
        self.viz_handlers_configs = viz_handlers_configs

        self.buffer_manager = buffer_manager

        self.decoder_head_type = decoder_head_type
        # TODO: remove this when we implement stitching / sliding window inference
        # self.feature_viz_type = feature_viz_type

        # self.scale_gt_boxes = scale_gt_boxes
        # self.gt_boxes_format = gt_boxes_format
        # self.pred_boxes_format = pred_boxes_format

        self.inference_save_dir = save_dir
        # TODO: remove this when we implement stitching / sliding window inference
        # self.inference_save_format = save_format
        # self.inference_zarr_chunk_shape = zarr_chunk_shape
        # self.inference_zarr_shard_shape = zarr_shard_shape

        # DB Related Configs
        # self.dbname = dbname
        # self.verbose = verbose
        # self.protocol = protocol
        # self.dotenv_path = dotenv_path
        # self.server_folder_path = server_folder_path
        # self._database_url = self._load_uri()

        # self.max_hypercubes = max_hypercubes
        # self.max_partitions = max_partitions

        assert outputs_metadata is not None, "outputs_metadata must be provided"
        # DictConfig -> plain dict[str, dict]
        self.outputs_metadata = {str(name): dict(meta) for name, meta in outputs_metadata.items()}

        # Normalize auxiliary_outputs into dict[name -> spec]
        if auxiliary_outputs is not None and not isinstance(auxiliary_outputs, (dict, list, tuple)):
            # Hydra DictConfig/ListConfig -> plain container
            auxiliary_outputs = OmegaConf.to_container(auxiliary_outputs, resolve=True)

        if auxiliary_outputs:
            if isinstance(auxiliary_outputs, dict):
                self.auxiliary_outputs = {str(name): dict(spec) for name, spec in auxiliary_outputs.items()}
            else:
                aux: Dict[str, Dict[str, Any]] = {}
                for item in auxiliary_outputs:
                    if isinstance(item, (tuple, list)) and len(item) == 2:
                        name, spec = item
                        aux[str(name)] = dict(spec)
                    else:
                        # item is expected to be a mapping with "name"
                        item = dict(item)
                        aux[str(item["name"])] = item
                self.auxiliary_outputs = aux
        else:
            self.auxiliary_outputs = {}

        self.save_auxiliary_outputs = bool(self.auxiliary_outputs)

        # FIXME: enforce user naming main_output_name instead
        # main prediction output name; assume first key in outputs_metadata
        self.main_output_name = next(iter(self.outputs_metadata.keys()))
        # self.num_output_channels = None
        # TODO: remove this when we implement stitching / sliding window inference
        # if self.stitch_volume:
        #     self.num_output_channels = self.outputs_metadata[self.main_output_name].get("num_output_channels")
        #     assert self.num_output_channels is not None, "num_output_channels must be specified for main output"

        # all data types we will aggregate/save:
        #   - main output (e.g. 'predictions')
        #   - plus each auxiliary output (e.g. 'data_tensor')
        # self.data_types = [self.main_output_name, *self.auxiliary_outputs.keys()]

        # self.prediction_df = self._get_data_tiles_metadata()
        os.makedirs(self.inference_save_dir, exist_ok=True)

        # TODO: remove this when we implement stitching / sliding window inference
        # if self.stitch_volume:
        #     self._build_state()
        # else:
        #     # no stitch -> no tile buffers
        #     self._tile_state, self._row_by_key, self._name_by_key = {}, {}, {}
        #     self._key_by_rank = {}
        #     self.rank, self.world_size = process_rank(), get_world_size()

        self.rank, self.world_size = process_rank(), get_world_size()

        # ray.logger.info(f"Inference Database: {self.prediction_df}")
        # ray.logger.info(f"Data types to save: {self.data_types}")
        ray.logger.info(f"Auxiliary outputs: {self.auxiliary_outputs}")
        ray.logger.info(f"Main output metadata: {self.outputs_metadata}")
        ray.logger.info(f"Aggregate mode: {self.aggregate_mode}")

    def _get_token_shape(self, token_shape: Tuple[int, ...], input_format: str) -> Tuple[int, ...]:
        if input_format == "ZYXC":
            return token_shape[1:-1]  # drop T and C dimensions
        elif input_format == "TZYXC":
            return token_shape[:-1] # drop C dimension
        else:
            raise ValueError(f"Unsupported input format: {input_format}")

    # TODO START: remove this below when we implement stitching / sliding window inference ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # def _get_data_tiles_metadata(self) -> pd.DataFrame:
    #     roi_csv = self.hypercubes_dataframe_path.with_name(f"{self.hypercubes_dataframe_path.stem}_rois.csv")
    #     if (not self.use_cached_hypercubes_dataframe) or (not roi_csv.exists()):
    #         query = self._get_query(roi_list=self.roi_list)
    #         table = self.execute_query(query)
    #     else:
    #         if self.verbose:
    #             print(f"Loading hypercubes dataframe from cached file: {roi_csv}")
    #         table = pd.read_csv(roi_csv)

    #         # FIXME: maintain uniform naming accross databases
    #         #        to avoid this
    #         cols_rename = [
    #             "tile_z_start",
    #             "tile_y_start",
    #             "tile_x_start",
    #             "tile_z_end",
    #             "tile_y_end",
    #             "tile_x_end",
    #             "tile_channel_size",
    #             "tile_time_size",
    #         ]
    #         if set(cols_rename).issubset(table.columns):
    #             table = table.rename(
    #                 columns={
    #                     "prepared_id": "id",
    #                     "tile_z_end": "z_end",
    #                     "tile_y_end": "y_end",
    #                     "tile_x_end": "x_end",
    #                     "tile_z_start": "z_start",
    #                     "tile_y_start": "y_start",
    #                     "tile_x_start": "x_start",
    #                     "tile_time_size": "time_size",
    #                     "tile_channel_size": "channel_size",
    #                 }
    #             )

    #         table = table[table["id"].isin(self.roi_list)]

    #         print(f"Loaded Table: {table}")

    #     roi_ids = table["id"].tolist()
    #     tiles = self._get_tiles_from_rois(roi_ids)
    #     table = table.merge(tiles, left_on="id", right_on="prepared_id", how="left")

    #     allowed = pd.DataFrame(self.roi_tile_list, columns=["id", "tile_name"]).drop_duplicates()
    #     table = table.merge(allowed, on=["id", "tile_name"], how="inner")

    #     # table = table.drop_duplicates(subset=["tile_name", "output_folder"], keep="first")

    #     table["prediction"] = [None] * len(table)
    #     table["count"] = [None] * len(table)

    #     return table

    # def _load_uri(self):
    #     if "SUPABASE_STAGING_URI" not in os.environ or "SUPABASE_PROD_URI" not in os.environ:
    #         assert Path(self.dotenv_path).exists(), f"{self.dotenv_path} was not found"
    #         if self.verbose:
    #             print(f"Loading additional environment variables from {self.dotenv_path}")
    #         load_dotenv(self.dotenv_path, verbose=True)

    #     if self.dbname == "staging":
    #         uri = os.environ.get("SUPABASE_STAGING_URI")
    #     elif self.dbname == "prod":
    #         uri = os.environ.get("SUPABASE_PROD_URI")
    #     else:
    #         raise ValueError(f"Unknown database name: {self.dbname}")

    #     assert uri is not None, "SUPABASE_URI_* environment variable not set"
    #     return uri

    # def execute_query(self, query: str | List[str]) -> pd.DataFrame:
    #     try:
    #         # avoid the costly COUNT query for pandas by using arrow as an intermediate step
    #         # https://sfu-db.github.io/connector-x/freq_questions.html
    #         result = cx.read_sql(
    #             conn=self._database_url,
    #             query=query,
    #             protocol=self.protocol,
    #             return_type="arrow",
    #         )
    #         df = result.to_pandas(split_blocks=False, date_as_object=False)
    #         return df
    #     except Exception as e:
    #         logger.error(f"Failed to execute query: {e}")
    #         raise

    # def _get_tiles_from_rois(
    #     self,
    #     roi_ids: list,
    #     table_name_shortcut: str = "hc",
    #     tile_list: Optional[list] = None,
    #     table_name: str = "prepared_tiles",
    #     idx_col: str = "prepared_id",
    # ) -> pd.DataFrame:
    #     query = f"""
    #         SELECT
    #             *
    #         FROM {table_name}
    #         WHERE {idx_col} IN ({', '.join(map(str, roi_ids))})
    #     """
    #     table = self.execute_query(query)
    #     if tile_list is not None:
    #         table = table[table["tile_name"].isin(tile_list)]
    #     return table

    # def _get_query(
    #     self,
    #     roi_list: list,
    #     tile_list: Optional[list] = None,
    #     column_names: list = [
    #         "id",
    #         "y_start",
    #         "x_start",
    #         "z_start",
    #         "y_end",
    #         "x_end",
    #         "z_end",
    #         "channel_size",
    #         "time_size",
    #         "output_folder",
    #     ],
    #     table_name: str = "prepared",
    #     table_name_shortcut: str = "hc",
    #     idx_col: str = "id",
    # ) -> str:
    #     filters = self._filters_to_string(
    #         table_name_shortcut=table_name_shortcut,
    #         roi_list=roi_list,
    #         tile_list=tile_list,
    #     )
    #     max_rows = self.count_rows(table_name=table_name)

    #     if self.max_hypercubes is None:
    #         max_hypercubes = max_rows
    #     else:
    #         max_hypercubes = self.max_hypercubes

    #     if max_hypercubes > max_rows:
    #         max_hypercubes = max_rows

    #     if max_hypercubes > 1000:
    #         # select max number of partitions that divides the number of rows in each partition evenly
    #         partition_num = (
    #             max([i for i in range(1, self.max_partitions + 1) if max_hypercubes % i == 0])
    #             if max_hypercubes is not None
    #             else 1
    #         )
    #         print(f"Using {partition_num} partitions to query.")
    #     else:
    #         partition_num = 1

    #     rows_per_partition = max_hypercubes // partition_num
    #     queries = [
    #         f"""
    #             SELECT
    #                 {', '.join([f'{table_name_shortcut}.{col}' for col in column_names])}
    #             FROM {table_name} {table_name_shortcut}
    #             {filters}
    #             ORDER BY {idx_col} DESC
    #             LIMIT {rows_per_partition}
    #             OFFSET {rows_per_partition * i}
    #         """
    #         for i in range(partition_num)
    #     ]

    #     return queries

    # def _filters_to_string(
    #     self,
    #     table_name_shortcut: str = "hc",
    #     max_rois: Optional[int] = None,
    #     max_tiles: Optional[int] = None,
    #     hpf_list: Optional[Iterable[int]] = None,
    #     roi_list: Optional[Iterable[int]] = None,
    #     tile_list: Optional[Iterable[str]] = None,
    # ) -> str:

    #     if self.server_folder_path is None or str(self.server_folder_path).startswith("/clusterfs"):
    #         filters = f"WHERE {table_name_shortcut}.exists IS TRUE"
    #     elif str(self.server_folder_path).startswith("/groups"):
    #         filters = f"WHERE {table_name_shortcut}.exists_prfs IS TRUE"
    #     elif str(self.server_folder_path).startswith("/aws") or str(self.server_folder_path).startswith(
    #         "/workspace/CellObservatoryData"
    #     ):
    #         filters = f"WHERE {table_name_shortcut}.exists_aws IS TRUE"
    #     elif str(self.server_folder_path).startswith("/lustre"):
    #         filters = f"WHERE {table_name_shortcut}.exists_oak IS TRUE"
    #     else:
    #         raise ValueError(f"Unknown server_folder_path: {self.server_folder_path}")

    #     if roi_list is not None or tile_list is not None:
    #         filters += self._choose_filter(rois=roi_list, tiles=tile_list, table_name=table_name_shortcut).replace(
    #             "WHERE", " AND "
    #         )
    #     elif max_rois is not None or max_tiles is not None:
    #         filters += self._limit_filter(
    #             max_rois=max_rois, max_tiles=max_tiles, table_name=table_name_shortcut
    #         ).replace("WHERE", " AND ")

    #     if hpf_list is not None:
    #         filters += self._age_filter(hpfs=hpf_list, table_name=table_name_shortcut).replace("WHERE", " AND ")

    #     if self.verbose:
    #         print(f"Using filters: {filters}")
    #     return filters

    # def _age_filter(self, hpfs: Iterable[int], table_name: str = "ptv") -> str:
    #     assert hpfs is not None, "hpfs must be provided"

    #     hpfs = tuple(hpfs) if len(hpfs) > 1 else f"({hpfs[0]})"
    #     return f"WHERE {table_name}.hpf IN {hpfs}"

    # def _limit_filter(
    #     self,
    #     max_rois: Optional[int] = None,
    #     max_tiles: Optional[int] = None,
    #     table_name: str = "ptv",
    #     idx_col: str = "id",
    # ) -> str:
    #     assert max_rois is not None or max_tiles is not None, "At least one of max_rois or max_tiles must be provided"

    #     if max_rois is not None:
    #         unique_rois = self.get_random_rois(max_rois)
    #         if isinstance(unique_rois, Iterable):
    #             filters = f"WHERE {table_name}.{idx_col} IN {tuple(unique_rois)}"
    #         else:
    #             filters = f"WHERE {table_name}.{idx_col} IN ('{unique_rois}')"
    #     else:
    #         if max_tiles > 1:
    #             unique_rois, unique_tiles = zip(*self.get_random_tiles(max_tiles))
    #         else:
    #             unique_rois, unique_tiles = self.get_random_tiles(max_tiles)

    #         if isinstance(unique_tiles, Iterable) and isinstance(unique_rois, Iterable):
    #             filters = (
    #                 f"WHERE {table_name}.{idx_col} IN {tuple(unique_rois)} "
    #                 f"AND {table_name}.tile_name IN {tuple(unique_tiles)}"
    #             )
    #         else:
    #             filters = (
    #                 f"WHERE {table_name}.{idx_col} IN ('{unique_rois}') "
    #                 f"AND {table_name}.tile_name IN ('{unique_tiles}')"
    #             )
    #     return filters

    # def _choose_filter(
    #     self,
    #     rois: Optional[Iterable[int | str]] = None,
    #     tiles: Optional[Iterable[str]] = None,
    #     table_name: str = "ptv",
    #     idx_col: str = "id",
    # ) -> str:

    #     def _sql_in_list(values):
    #         out = []
    #         for v in values:
    #             if isinstance(v, (int, float)) or (isinstance(v, str) and v.isnumeric()):
    #                 out.append(str(v))
    #             else:
    #                 out.append("'" + str(v).replace("'", "''") + "'")
    #         return "(" + ",".join(out) + ")"

    #     assert rois is not None or tiles is not None, "At least one of rois or tiles must be provided"

    #     clauses = []
    #     if rois is not None:
    #         rois_list = list(rois)
    #         clauses.append(f"{table_name}.{idx_col} IN {_sql_in_list(rois_list)}")

    #     if tiles is not None:
    #         tiles_list = list(tiles)
    #         clauses.append(f"{table_name}.tile_name IN {_sql_in_list(tiles_list)}")

    #     return "WHERE " + " AND ".join(clauses)

    # def count_rows(self, table_name: str) -> int:
    #     return self.execute_query(f"SELECT COUNT(*) FROM {table_name};").iloc[0, 0]
    # TODO END: remove this above when we implement stitching / sliding window inference ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _predict(self, batch_tensor: torch.Tensor, data_sample: dict) -> Dict[str, torch.Tensor]:
        if self.task == "detection":
            if self.decoder_head_type == "plaindetr":
                preds = self.model.predict(data_sample)
                if not isinstance(preds, dict):
                    if isinstance(preds, torch.Tensor):
                        preds = {self.main_output_name: preds}
                    else:
                        raise ValueError(f"Prediction returned unexpected type {type(preds)}")
            else:
                raise NotImplementedError(
                    f"Decoder head type {self.decoder_head_type} not supported for detection sliding window inference."
                )

        elif self.task == "instance_segmentation":
            if self.decoder_head_type == "maskdino":
                preds = self.model.predict(data_sample)
            elif self.decoder_head_type == "sam":
                preds = self.model.predict(data_sample, type="volume")
                preds, data_sample["data_tensor"] = postprocess_sam_preds(
                    preds, data_sample["data_tensor"]
                )
            else:
                raise NotImplementedError(
                    f"Decoder head type {self.decoder_head_type} not supported for instance segmentation sliding window inference."
                )

        elif self.task == "semantic_segmentation":
            raise NotImplementedError("Semantic segmentation decoder head not yet supported for sliding window inference.")            

        elif self.task == "dense_prediction":
            if self.task not in {"upsample_space", "upsample_time", "upsample_space_time", "channel_split"}:
                raise NotImplementedError(
                    f"Task {self.task} not implemented for {self.decoder_head_type} sliding window inference."
                )
            pred_hypercubes = self.model.predict(data_sample)
            preds = {self.main_output_name: pred_hypercubes}

        elif self.task == "pretrain":
            pred_hypercubes = self.model.predict(data_sample)
            preds = {self.main_output_name: pred_hypercubes}

        elif self.task == "feature_extractor":
            # NOTE: we expect patchified prediction tensors here
            pred_hypercubes = self.model.forward_features(batch_tensor, masks=None)
            
            if self.input_format == "ZYXC":
                B, N, C = pred_hypercubes.shape

                # FIXME: this will break if the backbone does further downsampling
                num = int(np.prod(self.token_shape))
                # CLS token
                if N == num + 1:
                    pred_hypercubes = pred_hypercubes[:, 1:, :]
                    N -= 1
                if N != num:
                    raise RuntimeError(f"forward_features returned N={N}, expected {num} (token_shape={self.token_shape})")

                pred_hypercubes = pred_hypercubes.view(B, *self.token_shape, C)
                pred_hypercubes = pred_hypercubes.unsqueeze(1)
            else:
                raise ValueError("Feature extractor only supports ZYXC input format for now.")

            preds = {self.main_output_name: pred_hypercubes}

        else:
            raise NotImplementedError(
                f"Decoder head type {self.decoder_head_type} not supported for sliding window inference."
            )

        return preds

    # TODO START: remove this below when we implement stitching / sliding window inference ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # def _build_state(self):
    #     self._tile_state, self._row_by_key, self._name_by_key = {}, {}, {}
    #     ws, rank = get_world_size(), process_rank()
    #     self._key_by_rank = {}

    #     for _, row in self.prediction_df.iterrows():
    #         roi = int(row["id"])
    #         name = str(row["tile_name"])
    #         owner = stable_key_owner(roi, name, ws)
    #         self._key_by_rank.setdefault(owner, []).append((roi, name))

    #     for roi, name in self._key_by_rank.get(rank, []):
    #         row = self.prediction_df[
    #             (self.prediction_df["id"] == roi) & (self.prediction_df["tile_name"] == name)
    #         ].iloc[0]

    #         vol_T = int(row["time_size"])
    #         vol_Z = int(row["z_end"] - row["z_start"])
    #         vol_Y = int(row["y_end"] - row["y_start"])
    #         vol_X = int(row["x_end"] - row["x_start"])
    #         out_spatial_shape = (vol_T, vol_Z, vol_Y, vol_X)

    #         tile_rows = self.database[
    #             (self.database["prepared_id"].astype(int) == roi) & (self.database["tile_name"].astype(str) == name)
    #         ]
    #         n_cubes = int(tile_rows.shape[0])

    #         z_size = int(tile_rows["z_size"].iloc[0])
    #         y_size = int(tile_rows["y_size"].iloc[0])
    #         x_size = int(tile_rows["x_size"].iloc[0])

    #         t_per_cube = (
    #             int(tile_rows["time_size"].iloc[0])
    #             if (self.input_format == "TZYXC" and "time_size" in tile_rows.columns)
    #             else 1
    #         )

    #         voxels_per_cube = t_per_cube * z_size * y_size * x_size
    #         tile_volume = n_cubes * voxels_per_cube

    #         key = (roi, tile_hash(name))
    #         st = {
    #             "row": row,
    #         }
    #         for dt in self.data_types:
    #             if dt == self.main_output_name:
    #                 C = self.num_output_channels

    #                 if self.task == "feature_extractor":
    #                     # for feature visualization, output shape is feature grid
    #                     # of tokens which are spatially smaller than the input tensor                        
    #                     Tt = ceil_div(vol_T, self.temporal_patch_size)
    #                     Zt = ceil_div(vol_Z, self.axial_patch_size)
    #                     Yt = ceil_div(vol_Y, self.lateral_patch_size)
    #                     Xt = ceil_div(vol_X, self.lateral_patch_size)
    #                     out_shape = (Tt, Zt, Yt, Xt, C)
    #                     tile_volume_dt = Tt * Zt * Yt * Xt

    #                 else:
    #                     out_shape = (*out_spatial_shape, C)
    #                     tile_volume_dt = tile_volume

    #             else:
    #                 meta = self.auxiliary_outputs.get(dt)
    #                 C = meta.get("num_output_channels")
    #                 assert C is not None, f"num_output_channels must be specified for auxiliary output {dt}"
    #                 tile_volume_dt = tile_volume
    #                 out_shape = (*out_spatial_shape, C)

    #             st[f"shape_{dt}"] = out_shape
    #             st[f"pred_{dt}"] = None
    #             st[f"cnt_{dt}"] = None
    #             st[f"remaining_{dt}"] = tile_volume_dt
    #             st[f"done_{dt}"] = False

    #         self._tile_state[key] = st
    #         self._row_by_key[key] = row
    #         self._name_by_key[key] = name

    #         ray.logger.info(f"Rank {rank} will process tile {name} (ROI {roi})")

    #     os.makedirs(self.inference_save_dir, exist_ok=True)

    # def _build_pred_buckets(self, pred_hypercubes: torch.Tensor, meta: dict, with_token_grid: bool):
    #     ws = get_world_size()
    #     pred_buckets = {r: [] for r in range(ws)}
    #     for b in range(pred_hypercubes.size(0)):
    #         roi = int(meta["prepared_id"][b])
    #         tile_nm = str(meta["tile_name"][b])
    #         owner_rank = stable_key_owner(roi, tile_nm, ws)

    #         t0 = int(meta["time_start"][b])
    #         T = int(meta["time_size"][b])
    #         t1 = t0 + T
    #         z0 = int(meta["z_start"][b])
    #         sz = int(meta["z_size"][b])
    #         z1 = z0 + sz
    #         y0 = int(meta["y_start"][b])
    #         sy = int(meta["y_size"][b])
    #         y1 = y0 + sy
    #         x0 = int(meta["x_start"][b])
    #         sx = int(meta["x_size"][b])
    #         x1 = x0 + sx

    #         patch = pred_hypercubes[b]
    #         if self.input_format == "ZYXC" and patch.ndim == 4:
    #             patch = patch.unsqueeze(0)

    #         if with_token_grid:
    #             st, sz, sy, sx = self.temporal_patch_size, self.axial_patch_size, self.lateral_patch_size, self.lateral_patch_size
    #             t0f, z0f, y0f, x0f = t0 // st, z0 // sz, y0 // sy, x0 // sx
    #             Tf, Zf, Yf, Xf = patch.shape[:4]  # patch is [Tt, Zt, Yt, Xt, C]
    #             coords = torch.tensor([t0f, t0f+Tf, z0f, z0f+Zf, y0f, y0f+Yf, x0f, x0f+Xf],
    #                                 device=patch.device, dtype=torch.int32)
    #         else:
    #             coords = torch.tensor([t0, t1, z0, z1, y0, y1, x0, x1], device=patch.device, dtype=torch.int32)

    #         pred_buckets[owner_rank].append(
    #             (
    #                 torch.tensor([roi, tile_hash(tile_nm)], device=patch.device, dtype=torch.long),
    #                 coords,
    #                 patch,
    #             )
    #         )

    #     return pred_buckets

    # def _pack_for_alltoall(
    #     self,
    #     chunks,  # List[(dst_rank, tensor)]
    #     world_size: int,
    #     device: torch.device,
    #     tail_shape: tuple,
    #     dtype: torch.dtype,
    # ):
    #     outs, splits = [], []
    #     for dst in range(world_size):
    #         found = next((x for x in chunks if x[0] == dst), None)
    #         if found is None:
    #             outs.append(torch.empty((0,) + tail_shape, device=device, dtype=dtype))
    #             splits.append(0)
    #         else:
    #             outs.append(found[1])
    #             splits.append(found[1].size(0))
    #     send = torch.cat(outs, dim=0) if outs else torch.empty((0,) + tail_shape, device=device, dtype=dtype)
    #     return send, splits

    # # TODO: this could be generalized to handle more types of payloads
    # def _alltoall(self, buckets, metadata: dict, out_channels: int):
    #     ws, rk = get_world_size(), process_rank()
    #     device = torch.device("cuda", torch.cuda.current_device())

    #     send_counts = torch.zeros(ws, dtype=torch.int32, device=device)
    #     keys_bucket, coords_bucket, pred_hypercubes_bucket = [], [], []

    #     for dst in range(ws):
    #         # bucket: {rank: [(key, coord, patch), ...], ...}
    #         bucket = buckets[dst]
    #         if not bucket:
    #             continue
    #         # keys: [N,2]
    #         pred_keys = torch.stack([t[0] for t in bucket], dim=0)
    #         # coords: [N,8]
    #         pred_coords = torch.stack([t[1] for t in bucket], dim=0)
    #         # patches: [N,T,S,S,S,C]
    #         pred_hypercube = torch.stack([t[2] for t in bucket], dim=0)
    #         keys_bucket.append((dst, pred_keys))
    #         coords_bucket.append((dst, pred_coords))
    #         pred_hypercubes_bucket.append((dst, pred_hypercube))
    #         send_counts[dst] = pred_keys.size(0)

    #     keys_send, keys_splits = self._pack_for_alltoall(
    #         keys_bucket, world_size=ws, device=device, tail_shape=(2,), dtype=torch.long
    #     )
    #     coords_send, coords_splits = self._pack_for_alltoall(
    #         coords_bucket, world_size=ws, device=device, tail_shape=(8,), dtype=torch.int32
    #     )

    #     # NOTE: assumes uniform hypercube shape with specific format
    #     if pred_hypercubes_bucket:
    #         tail = pred_hypercubes_bucket[0][1].shape[1:]  # (T,S,S,S,C)
    #     else:
    #         T = int(metadata["time_size"][0]) if "time_size" in metadata else 1
    #         Sz = int(metadata["z_size"][0])
    #         Sy = int(metadata["y_size"][0])
    #         Sx = int(metadata["x_size"][0])
    #         tail = (T, Sz, Sy, Sx, out_channels)

    #     pred_hypercubes_send, pred_hypercubes_splits = self._pack_for_alltoall(
    #         pred_hypercubes_bucket, world_size=ws, device=device, tail_shape=tail, dtype=self.dtype
    #     )

    #     send_counts_cuda = send_counts.to(device)
    #     recv_counts_cuda = torch.empty_like(send_counts_cuda)
    #     dist.all_to_all_single(output=recv_counts_cuda, input=send_counts_cuda)
    #     recv_counts = recv_counts_cuda.cpu().tolist()
    #     total_recv = sum(recv_counts)

    #     if keys_send is None:
    #         keys_recv = torch.empty((0, 2), device=device, dtype=torch.long)
    #         coords_recv = torch.empty((0, 8), device=device, dtype=torch.int32)
    #         pred_hypercubes_recv = (
    #             torch.empty((0,) + tail, device=device, dtype=pred_hypercubes_bucket[0][1].dtype) if tail else None
    #         )
    #     else:
    #         keys_recv = torch.empty((total_recv, 2), device=device, dtype=keys_send.dtype)
    #         coords_recv = torch.empty((total_recv, 8), device=device, dtype=coords_send.dtype)
    #         pred_hypercubes_recv = torch.empty((total_recv,) + tail, device=device, dtype=pred_hypercubes_send.dtype)

    #         dist.all_to_all_single(
    #             output=keys_recv, input=keys_send, output_split_sizes=recv_counts, input_split_sizes=keys_splits
    #         )
    #         dist.all_to_all_single(
    #             output=coords_recv, input=coords_send, output_split_sizes=recv_counts, input_split_sizes=coords_splits
    #         )
    #         dist.all_to_all_single(
    #             output=pred_hypercubes_recv,
    #             input=pred_hypercubes_send,
    #             output_split_sizes=recv_counts,
    #             input_split_sizes=pred_hypercubes_splits,
    #         )

    #     return keys_recv, coords_recv, pred_hypercubes_recv

    # def _apply_recv(self, keys, coords, pred_hypercubes, data_type: Optional[str] = None):
    #     N = keys.size(0)
    #     if N == 0:
    #         return set()

    #     if data_type is None:
    #         data_type = self.main_output_name

    #     done_keys = set()
    #     pred_hypercubes, keys, coords = pred_hypercubes.to("cpu"), keys.to("cpu"), coords.to("cpu")
    #     for i in range(N):
    #         roi_id = int(keys[i, 0].item())
    #         tile_h = int(keys[i, 1].item())
    #         key = (roi_id, tile_h)

    #         t0, t1, z0, z1, y0, y1, x0, x1 = coords[i].tolist()
    #         pred_hypercube = pred_hypercubes[i]

    #         pred_t, cnt_t = self._get_or_init_buffers(key, data_type=data_type)

    #         pred_view = pred_t[t0:t1, z0:z1, y0:y1, x0:x1, :]
    #         cnt_view = cnt_t[t0:t1, z0:z1, y0:y1, x0:x1, 0]

    #         T2, Z2, Y2, X2, C2 = pred_view.shape
    #         T, Z, Y, X, C = pred_hypercube.shape

    #         # if patch is larger (because of padding), crop it
    #         if (T != T2) or (Z != Z2) or (Y != Y2) or (X != X2):
    #             # we expect the patch to be >= view in each spatial dim
    #             if T < T2 or Z < Z2 or Y < Y2 or X < X2:
    #                 raise RuntimeError(
    #                     f"pred_hypercube smaller than pred_view: "
    #                     f"patch {pred_hypercube.shape}, view {pred_view.shape}"
    #                 )

    #             pred_hypercube = pred_hypercube[:T2, :Z2, :Y2, :X2, :]

    #         zeros_before = (cnt_view == 0).sum()

    #         if pred_hypercube.dtype != pred_view.dtype:
    #             pred_hypercube = pred_hypercube.to(pred_view.dtype)

    #         pred_view.add_(pred_hypercube)
    #         cnt_view.add_(1.0)

    #         zeros_after = (cnt_view == 0).sum()
    #         filled = int((zeros_before - zeros_after).item())
    #         self._tile_state[key][f"remaining_{data_type}"] -= filled

    #         # print(f"Tile state: {self._tile_state[key]["remaining_" + data_type]} remaining voxels for data type {data_type}")

    #         if all(self._tile_state[key][f"remaining_{dt}"] <= 0 for dt in self.data_types):
    #             # TODO: this should most likely happen on a separate Actor
    #             #       to avoid blocking the main inference loop for saving files
    #             if self._finish_if_done(key):
    #                 done_keys.add(key)

    #     return done_keys

    # def _finish_if_done(self, key, force: bool = False):
    #     st = self._tile_state[key]
    #     if any(st[f"done_{dt}"] or st[f"pred_{dt}"] is None for dt in self.data_types):
    #         return False
    #     if not force and any(st[f"remaining_{dt}"] != 0 for dt in self.data_types):
    #         return False

    #     row = self._row_by_key[key]
    #     name = self._name_by_key[key]
    #     try:
    #         base = str(row["output_folder"])
    #     except KeyError:
    #         base = f"inference_roi{row.get('id', 'unknown')}"

    #     base_sample_name = base.replace("/", "_") + "_" + name
    #     base_sample_name = base_sample_name.replace(".zarr", "").replace(".tiff", "")

    #     preds_dict = {}
    #     for dt in self.data_types:
    #         p = st[f"pred_{dt}"]
    #         c = st[f"cnt_{dt}"]

    #         c.clamp_min_(1.0)
    #         p.div_(c)
    #         preds = p

    #         # Optional per-output activation from outputs_metadata
    #         meta = self.outputs_metadata.get(dt) or self.auxiliary_outputs.get(dt) or {}
    #         activation = meta.get("activation", None)
    #         if activation is not None:
    #             preds = activation(preds)

    #         if self.timepoint_list is not None:
    #             preds = preds[self.timepoint_list, ...]

    #         preds_dict[dt] = preds

    #     if self.task == "feature_extractor":
    #         handler_config = self.viz_handlers_configs.get("feature_viz")
    #         if handler_config is None:
    #             raise ValueError("feature_viz handler config is not provided")
    #         handler_config.update(
    #             gt_key="data_tensor",
    #             feat_key=self.main_output_name,
    #             z_step_pdf=self.z_step_pdf,
    #             pmin=self.pmin,
    #             pmax=self.pmax,
    #             fit="per_t",
    #             sample_voxels=50_000,
    #             viz=self.feature_viz_type,
    #             stride_zyx=(self.axial_patch_size, self.lateral_patch_size, self.lateral_patch_size)
    #         )
    #         self.viz_worker.visualize.remote(
    #             inference_outputs=preds_dict,
    #             save_dir=self.inference_save_dir,
    #             handler_configs=self.viz_handlers_configs,
    #         )
    #         print(f"Finished saving feature visualizations for tile {name} (ROI {row.get('id', 'unknown')})")
    #     else:
    #         handler_config = self.viz_handlers_configs.get("save_predictions")
    #         if handler_config is None:
    #             raise ValueError("save_predictions handler config is not provided")
    #         handler_config.update(
    #             {"name":base_sample_name},
    #         )
    #         self.viz_worker.visualize.remote(
    #             inference_outputs=preds_dict,
    #             save_dir=self.inference_save_dir,
    #             handler_configs=self.viz_handlers_configs,
    #         )

    #         print(f"Finished saving predictions for tile {name} (ROI {row.get('id', 'unknown')})")
    #         print(
    #             f"[finish_if_done] key={key}, base_sample_name={base_sample_name}, "
    #             f"save_dir={self.inference_save_dir}, vizualize_outputs={self.vizualize_outputs}, "
    #             f"save_outputs={self.save_outputs}"
    #         )

    #     for dt in self.data_types:
    #         st[f"done_{dt}"] = True

    #     return True

    # def _get_or_init_buffers(self, key, data_type: str):
    #     st = self._tile_state[key]
    #     if st[f"pred_{data_type}"] is None:
    #         st[f"pred_{data_type}"] = torch.zeros(st[f"shape_{data_type}"], dtype=self.dtype, device="cpu")
    #         st[f"cnt_{data_type}"] = torch.zeros((*st[f"shape_{data_type}"][:-1], 1), dtype=torch.float16, device="cpu")
    #     return st[f"pred_{data_type}"], st[f"cnt_{data_type}"]
    # TODO END: remove this above when we implement stitching / sliding window inference ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _PATH_TOKEN = re.compile(r"""
        ([^.[]+)
        (?:\[(\d+)\])?
    """, re.X)

    @staticmethod
    def resolve_path(root: Any, path: str) -> Any:
        """
        Resolve 'path' against 'root'.
        Supports dict keys, attributes, and list indices via '[i]' or bare '.i'.
        Examples:
        - "data_tensor"
        - "metainfo.masks[0]"
        - "metainfo.masks.0"   (treated like index 0)
        """
        cur = root
        for part in path.split("."):
            if part == "":
                continue

            # allow bare numeric segments as list indices (".0")
            if part.isdigit():
                cur = cur[int(part)]
                continue

            # support "key" and "key[i]"
            m = InferencerWorker._PATH_TOKEN.fullmatch(part)
            if not m:
                raise KeyError(f"Bad path segment: {part!r} (full path: {path!r})")
            key, idx = m.group(1), m.group(2)

            # descend by key/attr
            if isinstance(cur, dict):
                cur = cur[key]
            else:
                cur = getattr(cur, key)

            # optional index
            if idx is not None:
                cur = cur[int(idx)]

        return cur

    # TODO: remove this when we implement stitching / sliding window inference
    # @staticmethod
    # def _iter_aux_specs(auxiliary_outputs: Any) -> Iterable[Tuple[str, Dict[str, Any]]]:
    #     """
    #     Yields (name, spec) pairs from either:
    #     - list of {name, path, ...}
    #     - dict name -> {path, ...}
    #     """
    #     if auxiliary_outputs is None:
    #         return []
    #     if isinstance(auxiliary_outputs, dict):
    #         return auxiliary_outputs.items()
    #     # assume list-like
    #     return ((spec["name"], spec) for spec in auxiliary_outputs)

    def _preprocess(self, data_sample: dict) -> dict:
        """
        Materialize + normalize all outputs that will be saved
        """
        specs: Dict[str, Dict[str, Any]] = {}

        for name, meta in self.outputs_metadata.items():
            m = dict(meta) if meta is not None else {}
            specs[name] = m

        for aux_name, spec in self.auxiliary_outputs.items():
            s = dict(spec) if spec is not None else {}
            specs[aux_name] = s

        for name, spec in specs.items():
            path = spec.get("path", name)

            try:
                x = self.resolve_path(data_sample, path)
            except Exception:
                continue  # predictions etc not in data_sample yet

            if not isinstance(x, torch.Tensor):
                raise TypeError(f"{name}: expected torch.Tensor at preprocessing, got {type(x)} (path={path!r})")

            if bool(spec.get("patchified", False)):
                x = self.pe_unpatchify(x, out_channels=spec.get("num_output_channels"))

            data_sample[name] = x

        return data_sample

    def predict(self, data_sample: dict):
        """
        Predict and save inference outputs for a single data sample
        """
        if self.inference_mode == "full_tile" or self.inference_mode == "hypercube":
            if self.aggregate_mode != "none":
                ray.logger.warning("Full tile inference does not support aggregation.")

            # TODO: Double check if we need to do this
            data_sample = self._preprocess(data_sample)

            X = data_sample["data_tensor"]
            metadata = data_sample["metainfo"]

            preds = self._predict(X, data_sample)

            # targets may be absent in pure inference; normalize to per-record list[dict]
            B = len(metadata["prepared_id"])
            targets = metadata.get("targets", None)
            if targets is None:
                targets = [{} for _ in range(B)]
            elif isinstance(targets, dict):
                targets = [targets for _ in range(B)]
            elif isinstance(targets, (list, tuple)) and len(targets) == 1 and isinstance(targets[0], (list, tuple)):
                # common pattern: targets wrapped once
                targets = targets[0]
            self._save_inference_outputs(
                data_sample=data_sample,
                preds=preds,
            )

        elif self.inference_mode == "sliding_window":
            raise NotImplementedError("Sliding window inference is not implemented yet.")
            # TODO: remove this when we implement stitching / sliding window inference
            # if self.aggregate_mode == "stitch_volume":
            #     data_sample = self._preprocess(data_sample)

            # X = data_sample["data_tensor"]
            # metadata = data_sample["metainfo"]

            # preds = self._predict(X, data_sample)

            # if self.aggregate_mode == "stitch_volume":
            #     raise NotImplementedError("Stitching volume is not implemented for this task.")
            #     pred_hypercubes = preds[self.main_output_name]
            #     if not isinstance(pred_hypercubes, torch.Tensor):
            #         raise TypeError(
            #             f"stitch_volume requires Tensor preds[{self.main_output_name!r}], got {type(pred_hypercubes)}"
            #         )

            #     buckets = self._build_pred_buckets(
            #         pred_hypercubes,
            #         metadata,
            #         with_token_grid=(self.task == "feature_extractor"),
            #     )
            #     keys_recv, coords_recv, pred_hypercubes_recv = self._alltoall(
            #         buckets, metadata, out_channels=self.num_output_channels
            #     )
            #     self._apply_recv(keys_recv, coords_recv, pred_hypercubes_recv, data_type=self.main_output_name)

            #     if self.save_auxiliary_outputs:
            #         for aux_name, spec in self._iter_aux_specs(self.auxiliary_outputs):
            #             out_ch = spec["num_output_channels"]
            #             aux_pred_hypercubes = data_sample[aux_name]
            #             aux_buckets = self._build_pred_buckets(aux_pred_hypercubes, metadata, with_token_grid=False)
            #             aux_keys_recv, aux_coords_recv, aux_pred_hypercubes_recv = self._alltoall(
            #                 aux_buckets, metadata, out_channels=out_ch
            #             )
            #             self._apply_recv(aux_keys_recv, aux_coords_recv, aux_pred_hypercubes_recv, data_type=aux_name)

            # elif self.aggregate_mode == "none":
            #     # targets may be absent in pure inference; normalize to per-record list[dict]
            #     B = len(metadata["prepared_id"])
            #     targets = metadata.get("targets", None)
            #     if targets is None:
            #         targets = [{} for _ in range(B)]
            #     elif isinstance(targets, dict):
            #         targets = [targets for _ in range(B)]
            #     elif isinstance(targets, (list, tuple)) and len(targets) == 1 and isinstance(targets[0], (list, tuple)):
            #         # common pattern: targets wrapped once
            #         targets = targets[0]
            #     self._save_inference_outputs(
            #         data_sample=data_sample,
            #         preds=preds,
            #     )

            # else:
            #     raise ValueError(f"Unknown aggregate_mode: {self.aggregate_mode}")
        

    def _to_cpu_detached(self, x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        if isinstance(x, dict):
            return {k: self._to_cpu_detached(v) for k, v in x.items()}
        if isinstance(x, list):
            return [self._to_cpu_detached(v) for v in x]
        if isinstance(x, tuple):
            return tuple(self._to_cpu_detached(v) for v in x)
        return x

    def _to_cpu_detached(self, x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        if isinstance(x, dict):
            return {k: self._to_cpu_detached(v) for k, v in x.items()}
        if isinstance(x, list):
            return [self._to_cpu_detached(v) for v in x]
        if isinstance(x, tuple):
            return tuple(self._to_cpu_detached(v) for v in x)
        return x

    def _prepare_outputs_for_saving(self, data_sample: dict, preds: dict) -> Tuple[dict, dict]:
        """
        Prepare data sample for saving
        """
        if hasattr(self.model, "prepare_outputs_for_saving"):
            return self.model.prepare_outputs_for_saving(data_sample, preds)
        else:
            return data_sample, preds

    def _prepare_outputs_for_visualization(self, data_sample: dict, preds: dict) -> Tuple[dict, dict]:
        """
        Prepare data sample for visualization
        """
        if hasattr(self.model, "prepare_outputs_for_visualization"):
            return self.model.prepare_outputs_for_visualization(data_sample, preds)
        else:
            return data_sample, preds

    def _should_visualize(self, data_sample: dict, preds: dict) -> bool:
        """
        Check if the data sample should be visualized
        """
        if self.viz_sampling_policy is None:
            return True
        if self.viz_sampling_policy["name"] == "by_tile":
            return data_sample["metainfo"]["tile_name"] in self.viz_sampling_policy["tile_names"]
        if self.viz_sampling_policy["name"] == "random_sample":
            return np.random.rand() < self.viz_sampling_policy["fraction"]
        return False

    def _save_inference_outputs(self, data_sample: dict, preds: dict) -> None:
        """
        Prepare data sample for saving
        """
        if self.save_outputs:
            data_sample, preds = self._prepare_outputs_for_saving(data_sample, preds)
        if self.vizualize_outputs:
            data_sample, preds = self._prepare_outputs_for_visualization(data_sample, preds)
        try:
            sample_metainfo: Dict[str, Any] = data_sample["metainfo"]
        except KeyError:
            raise ValueError("data_sample must contain a 'metainfo' key")
        # TODO: Move targets to outer key rather than in metadata
        # That way the dict has all heavy elements at the outer level
        # and we can freely pass metadata via IPC without worrying 
        # about serialization of huge tensors
        targets = sample_metainfo.pop("targets", None) 
        save_outputs = {
            "metainfo": sample_metainfo.copy(),
        }
        viz_outputs = {
            "metainfo": sample_metainfo.copy(),
        }
        if targets is not None:
            sample_metainfo["targets"] = targets
            data_sample["targets"] = targets

        for output_name in self.outputs_metadata["save_tensors"]:
            if output_name in preds.keys():
                output_tensor = preds[output_name]
            elif output_name in data_sample.keys():
                output_tensor = data_sample[output_name]
            else:
                raise ValueError(f"Tensor {output_name} not found in preds or data_sample")
            
            if output_tensor is None:
                continue

            save_buffer = self.buffer_manager.get_buffer(f"{output_name}_save")
            if self.block_on_save:
                slot_info = ray.get(save_buffer.get_free.remote())
                if slot_info is None:
                    raise RuntimeError(f"No free slot found for {output_name}")
            else:
                slot_info = ray.get(save_buffer.try_get_free.remote())

            if slot_info is not None:
                dest_array = slot_info_to_view(slot_info)
                dest_array[:] = preds[output_name].cpu().numpy()
                save_outputs[output_name] = slot_info
        
        if self.save_outputs:
            if self.save_worker is None:
                raise RuntimeError("Attempting to save outputs but save_worker is None")
            self.save_worker.save.remote(
                inference_outputs=save_outputs,
                save_mode=self.save_mode,
                save_dir=self.inference_save_dir,
            )

        should_visualize = self._should_visualize(data_sample, preds)
        if not should_visualize:
            return

        for output_name in self.outputs_metadata["visualize_tensors"]:
            if output_name in preds.keys():
                output_tensor = preds[output_name]
            elif output_name in data_sample.keys():
                output_tensor = data_sample[output_name]
            else:
                raise ValueError(f"Output {output_name} not found in preds or data_sample")
            
            if output_tensor is None:
                continue
            
            viz_buffer = self.buffer_manager.get_buffer(f"{output_name}_viz")
            if self.block_on_viz:
                slot_info = ray.get(viz_buffer.get_free.remote())
                if slot_info is None:
                    raise RuntimeError(f"No free slot found for {output_name}")
            else:
                slot_info = ray.get(viz_buffer.try_get_free.remote())

            if slot_info is not None:
                dest_array = slot_info_to_view(slot_info)
                dest_array[:] = output_tensor.cpu().numpy()
                viz_outputs[output_name] = slot_info

        if self.vizualize_outputs:
            if self.viz_worker is None:
                raise RuntimeError("Attempting to visualize outputs but viz_worker is None")
            self.viz_worker.visualize.remote(
                inference_outputs=viz_outputs,
                save_dir=self.inference_save_dir,
                handler_configs=self.viz_handlers_configs,
            )

    def get_step_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for the current step
        """
        buffer_metrics = self.buffer_manager.get_metrics()
        metrics = {}
        for pool_name, pool_metrics in buffer_metrics.items():
            metrics[f"{pool_name}_avg_get_free_wait_time_s"] = pool_metrics["get_free_wait_time_s"] / pool_metrics["get_free_count"]
            metrics[f"{pool_name}_avg_put_free_wait_time_s"] = pool_metrics["put_free_wait_time_s"] / pool_metrics["put_free_count"]
            metrics[f"{pool_name}_avg_try_get_free_wait_time_s"] = pool_metrics["try_get_free_wait_time_s"] / pool_metrics["try_get_free_count"]
            if pool_metrics["try_get_free_count"] > 0:
                metrics[f"{pool_name}_pct_try_get_free_drops"] = 100 * pool_metrics["try_get_free_drops"] / pool_metrics["try_get_free_count"]
            else:
                metrics[f"{pool_name}_pct_try_get_free_drops"] = 0
            if pool_metrics["capacity"] > 0:
                metrics[f"{pool_name}_pct_in_use_current"] = 100 * pool_metrics["in_use_current"] / pool_metrics["capacity"]
            else:
                metrics[f"{pool_name}_pct_in_use_current"] = 0

        save_metrics = ray.get(self.save_worker.get_metrics.remote())
        viz_metrics = ray.get(self.viz_worker.get_metrics.remote())
        save_total = save_metrics["save_successes"] + save_metrics["save_failures"]
        viz_total = viz_metrics["visualize_successes"] + viz_metrics["visualize_failures"]
        metrics["save_worker_avg_save_time_ms"] = save_metrics["save_time_ms"] / save_total
        metrics["save_worker_pct_save_successes"] = 100 * save_metrics["save_successes"] / save_total
        metrics["save_worker_pct_save_failures"] = 100 * save_metrics["save_failures"] / save_total
        metrics["viz_worker_avg_visualize_time_ms"] = viz_metrics["visualize_time_ms"] / viz_total
        metrics["viz_worker_pct_visualize_successes"] = 100 * viz_metrics["visualize_successes"] / viz_total
        metrics["viz_worker_pct_visualize_failures"] = 100 * viz_metrics["visualize_failures"] / viz_total
        return metrics

    def finalize(self):
        # TODO: remove this when we implement stitching / sliding window inference
        # if self.stitch_volume:
        #     for key in list[Any](self._tile_state.keys()):
        #         self._finish_if_done(key, force=True)
        barrier()




# class InferencerWorker:
#     def __init__(
#         self,
#         aggregate_mode: Literal["stitch_volume", "save_local"],
#         task: str,
#         outputs_metadata: dict,
#         model: torch.nn.Module,
#         database: pd.DataFrame,
#         use_cached_hypercubes_dataframe: bool,
#         hypercubes_dataframe_path: Path,
#         timepoint_list: Optional[List[int]],
#         vizualize_outputs: bool,
#         save_as_volume: bool,
#         z_step_pdf: int,
#         input_format: str,
#         input_shape: List[int],
#         patch_shape: List[Optional[int]],
#         decoder_head_type: str,
#         roi_tile_list: List[Tuple[int, str]],
#         save_dir: Path | str,
#         dtype: torch.dtype = torch.float16,
#         protocol: Literal["binary", "csv", "cursor"] = "binary",
#         dotenv_path: Optional[Path] = Path(__file__).parent.parent.parent / ".env",
#         dbname: Literal["staging", "prod"] = "prod",
#         server_folder_path: Optional[Path] = None,
#         verbose: bool = False,
#         max_hypercubes: Optional[int] = None,
#         max_partitions: int = 10,
#         save_format: Literal["tiff", "zarr"] = "tiff",
#         zarr_chunk_shape: Optional[Tuple[int, ...]] = None,
#         zarr_shard_shape: Optional[Tuple[int, ...]] = None,
#         auxiliary_outputs: Optional[Any] = None,
#         pmin: float = 1.0,
#         pmax: float = 99.0,
#         feature_viz_type: Optional[str] = None,
#         pred_boxes_format: Literal["xyzxyz", "cxcyczwhd"] = "xyzxyz",
#         gt_boxes_format: Literal["xyzxyz", "cxcyczwhd"] = "cxcyczwhd",
#         scale_gt_boxes: bool = True,
#     ):
#         self.database = database
#         self.hypercubes_dataframe_path = Path(hypercubes_dataframe_path)
#         self.use_cached_hypercubes_dataframe = use_cached_hypercubes_dataframe
#         # TODO: consider alternative methods for only performing inference
#         #       on a subset of timepoints without storing a prediction for all timepoints
#         self.timepoint_list = timepoint_list

#         self.dtype = dtype
#         self.input_shape = input_shape
#         self.patch_shape = patch_shape
#         self.input_format = input_format

#         temporal_patch_size, self.axial_patch_size, self.lateral_patch_size = get_patch_sizes(
#             input_format=self.input_format, patch_shape=patch_shape
#         )
#         _, token_shape = calc_num_patches(
#             input_fmt=self.input_format,
#             input_shape=self.input_shape,
#             patch_shape=patch_shape,
#         )
#         self.pe_unpatchify = functools.partial(
#             PatchEmbedding.unpatchify,
#             temporal_patch_size=temporal_patch_size,
#             axial_patch_size=self.axial_patch_size,
#             lateral_patch_size=self.lateral_patch_size,
#             token_shape=token_shape,
#             input_format=self.input_format,
#         )
#         # NOTE: if input format does not contain 'T' we set patch size to 1
#         #       since buffers assume that the T-axis exists
#         self.temporal_patch_size = temporal_patch_size if temporal_patch_size else 1
#         self.token_shape = self._get_token_shape(token_shape, self.input_format)

#         # roi_tile_list is a list of (roi_id, tile_name) tuples
#         # to restrict inference to
#         self.roi_tile_list = roi_tile_list
#         self.roi_list = list(set([x[0] for x in roi_tile_list]))
#         self.tile_list = list(set([x[1] for x in roi_tile_list]))

#         self.model = model

#         self.task = task
#         self.aggregate_mode = aggregate_mode
#         if self.aggregate_mode == "save_local":
#             assert self.task in {"detection", "instance_segmentation", "semantic_segmentation"}, \
#                 "save_local aggregate_mode only supported for detection and segmentation tasks"

#         self.stitch_volume = (self.aggregate_mode == "stitch_volume")

#         self.pmin = pmin
#         self.pmax = pmax
#         self.z_step_pdf = z_step_pdf
#         self.save_as_pdf = save_as_pdf
#         self.save_as_volume = save_as_volume

#         self.decoder_head_type = decoder_head_type
#         self.feature_viz_type = feature_viz_type

#         self.scale_gt_boxes = scale_gt_boxes
#         self.gt_boxes_format = gt_boxes_format
#         self.pred_boxes_format = pred_boxes_format

#         self.inference_save_dir = save_dir
#         self.inference_save_format = save_format
#         self.inference_zarr_chunk_shape = zarr_chunk_shape
#         self.inference_zarr_shard_shape = zarr_shard_shape

#         self.dbname = dbname
#         self.verbose = verbose
#         self.protocol = protocol
#         self.dotenv_path = dotenv_path
#         self.server_folder_path = server_folder_path
#         self._database_url = self._load_uri()

#         self.max_hypercubes = max_hypercubes
#         self.max_partitions = max_partitions

#         assert outputs_metadata is not None, "outputs_metadata must be provided"
#         # DictConfig -> plain dict[str, dict]
#         self.outputs_metadata = {str(name): dict(meta) for name, meta in outputs_metadata.items()}

#         # Normalize auxiliary_outputs into dict[name -> spec]
#         if auxiliary_outputs is not None and not isinstance(auxiliary_outputs, (dict, list, tuple)):
#             # Hydra DictConfig/ListConfig -> plain container
#             auxiliary_outputs = OmegaConf.to_container(auxiliary_outputs, resolve=True)

#         if auxiliary_outputs:
#             if isinstance(auxiliary_outputs, dict):
#                 self.auxiliary_outputs = {str(name): dict(spec) for name, spec in auxiliary_outputs.items()}
#             else:
#                 aux: Dict[str, Dict[str, Any]] = {}
#                 for item in auxiliary_outputs:
#                     if isinstance(item, (tuple, list)) and len(item) == 2:
#                         name, spec = item
#                         aux[str(name)] = dict(spec)
#                     else:
#                         # item is expected to be a mapping with "name"
#                         item = dict(item)
#                         aux[str(item["name"])] = item
#                 self.auxiliary_outputs = aux
#         else:
#             self.auxiliary_outputs = {}

#         self.save_auxiliary_outputs = bool(self.auxiliary_outputs)

#         # FIXME: enforce user naming main_output_name instead
#         # main prediction output name; assume first key in outputs_metadata
#         self.main_output_name = next(iter(self.outputs_metadata.keys()))
#         self.num_output_channels = None
#         if self.stitch_volume:
#             self.num_output_channels = self.outputs_metadata[self.main_output_name].get("num_output_channels")
#             assert self.num_output_channels is not None, "num_output_channels must be specified for main output"

#         # all data types we will aggregate/save:
#         #   - main output (e.g. 'predictions')
#         #   - plus each auxiliary output (e.g. 'data_tensor')
#         self.data_types = [self.main_output_name, *self.auxiliary_outputs.keys()]

#         self.prediction_df = self._get_data_tiles_metadata()
#         os.makedirs(self.inference_save_dir, exist_ok=True)

#         if self.stitch_volume:
#             self._build_state()
#         else:
#             # no stitch -> no tile buffers
#             self._tile_state, self._row_by_key, self._name_by_key = {}, {}, {}
#             self._key_by_rank = {}
#             self.rank, self.world_size = process_rank(), get_world_size()

#         ray.logger.info(f"Inference Database: {self.prediction_df}")
#         ray.logger.info(f"Data types to save: {self.data_types}")
#         ray.logger.info(f"Auxiliary outputs: {self.auxiliary_outputs}")
#         ray.logger.info(f"Main output metadata: {self.outputs_metadata}")
#         ray.logger.info(f"Aggregate mode: {self.aggregate_mode}")

#     def _get_token_shape(self, token_shape: Tuple[int, ...], input_format: str) -> Tuple[int, ...]:
#         if input_format == "ZYXC":
#             return token_shape[1:-1]  # drop T and C dimensions
#         else:
#             raise ValueError(f"Unsupported input format: {input_format}")

#     def _get_data_tiles_metadata(self) -> pd.DataFrame:
#         roi_csv = self.hypercubes_dataframe_path.with_name(f"{self.hypercubes_dataframe_path.stem}_rois.csv")
#         if (not self.use_cached_hypercubes_dataframe) or (not roi_csv.exists()):
#             query = self._get_query(roi_list=self.roi_list)
#             table = self.execute_query(query)
#         else:
#             if self.verbose:
#                 print(f"Loading hypercubes dataframe from cached file: {roi_csv}")
#             table = pd.read_csv(roi_csv)

#             # FIXME: maintain uniform naming accross databases
#             #        to avoid this
#             cols_rename = [
#                 "tile_z_start",
#                 "tile_y_start",
#                 "tile_x_start",
#                 "tile_z_end",
#                 "tile_y_end",
#                 "tile_x_end",
#                 "tile_channel_size",
#                 "tile_time_size",
#             ]
#             if set(cols_rename).issubset(table.columns):
#                 table = table.rename(
#                     columns={
#                         "prepared_id": "id",
#                         "tile_z_end": "z_end",
#                         "tile_y_end": "y_end",
#                         "tile_x_end": "x_end",
#                         "tile_z_start": "z_start",
#                         "tile_y_start": "y_start",
#                         "tile_x_start": "x_start",
#                         "tile_time_size": "time_size",
#                         "tile_channel_size": "channel_size",
#                     }
#                 )

#             table = table[table["id"].isin(self.roi_list)]

#             print(f"Loaded Table: {table}")

#         roi_ids = table["id"].tolist()
#         tiles = self._get_tiles_from_rois(roi_ids)
#         table = table.merge(tiles, left_on="id", right_on="prepared_id", how="left")

#         allowed = pd.DataFrame(self.roi_tile_list, columns=["id", "tile_name"]).drop_duplicates()
#         table = table.merge(allowed, on=["id", "tile_name"], how="inner")

#         # table = table.drop_duplicates(subset=["tile_name", "output_folder"], keep="first")

#         table["prediction"] = [None] * len(table)
#         table["count"] = [None] * len(table)

#         return table

#     def _load_uri(self):
#         if "SUPABASE_STAGING_URI" not in os.environ or "SUPABASE_PROD_URI" not in os.environ:
#             assert Path(self.dotenv_path).exists(), f"{self.dotenv_path} was not found"
#             if self.verbose:
#                 print(f"Loading additional environment variables from {self.dotenv_path}")
#             load_dotenv(self.dotenv_path, verbose=True)

#         if self.dbname == "staging":
#             uri = os.environ.get("SUPABASE_STAGING_URI")
#         elif self.dbname == "prod":
#             uri = os.environ.get("SUPABASE_PROD_URI")
#         else:
#             raise ValueError(f"Unknown database name: {self.dbname}")

#         assert uri is not None, "SUPABASE_URI_* environment variable not set"
#         return uri

#     def execute_query(self, query: str | List[str]) -> pd.DataFrame:
#         try:
#             # avoid the costly COUNT query for pandas by using arrow as an intermediate step
#             # https://sfu-db.github.io/connector-x/freq_questions.html
#             result = cx.read_sql(
#                 conn=self._database_url,
#                 query=query,
#                 protocol=self.protocol,
#                 return_type="arrow",
#             )
#             df = result.to_pandas(split_blocks=False, date_as_object=False)
#             return df
#         except Exception as e:
#             logger.error(f"Failed to execute query: {e}")
#             raise

#     def _get_tiles_from_rois(
#         self,
#         roi_ids: list,
#         table_name_shortcut: str = "hc",
#         tile_list: Optional[list] = None,
#         table_name: str = "prepared_tiles",
#         idx_col: str = "prepared_id",
#     ) -> pd.DataFrame:
#         query = f"""
#             SELECT
#                 *
#             FROM {table_name}
#             WHERE {idx_col} IN ({', '.join(map(str, roi_ids))})
#         """
#         table = self.execute_query(query)
#         if tile_list is not None:
#             table = table[table["tile_name"].isin(tile_list)]
#         return table

#     def _get_query(
#         self,
#         roi_list: list,
#         tile_list: Optional[list] = None,
#         column_names: list = [
#             "id",
#             "y_start",
#             "x_start",
#             "z_start",
#             "y_end",
#             "x_end",
#             "z_end",
#             "channel_size",
#             "time_size",
#             "output_folder",
#         ],
#         table_name: str = "prepared",
#         table_name_shortcut: str = "hc",
#         idx_col: str = "id",
#     ) -> str:
#         filters = self._filters_to_string(
#             table_name_shortcut=table_name_shortcut,
#             roi_list=roi_list,
#             tile_list=tile_list,
#         )
#         max_rows = self.count_rows(table_name=table_name)

#         if self.max_hypercubes is None:
#             max_hypercubes = max_rows
#         else:
#             max_hypercubes = self.max_hypercubes

#         if max_hypercubes > max_rows:
#             max_hypercubes = max_rows

#         if max_hypercubes > 1000:
#             # select max number of partitions that divides the number of rows in each partition evenly
#             partition_num = (
#                 max([i for i in range(1, self.max_partitions + 1) if max_hypercubes % i == 0])
#                 if max_hypercubes is not None
#                 else 1
#             )
#             print(f"Using {partition_num} partitions to query.")
#         else:
#             partition_num = 1

#         rows_per_partition = max_hypercubes // partition_num
#         queries = [
#             f"""
#                 SELECT
#                     {', '.join([f'{table_name_shortcut}.{col}' for col in column_names])}
#                 FROM {table_name} {table_name_shortcut}
#                 {filters}
#                 ORDER BY {idx_col} DESC
#                 LIMIT {rows_per_partition}
#                 OFFSET {rows_per_partition * i}
#             """
#             for i in range(partition_num)
#         ]

#         return queries

#     def _filters_to_string(
#         self,
#         table_name_shortcut: str = "hc",
#         max_rois: Optional[int] = None,
#         max_tiles: Optional[int] = None,
#         hpf_list: Optional[Iterable[int]] = None,
#         roi_list: Optional[Iterable[int]] = None,
#         tile_list: Optional[Iterable[str]] = None,
#     ) -> str:

#         if self.server_folder_path is None or str(self.server_folder_path).startswith("/clusterfs"):
#             filters = f"WHERE {table_name_shortcut}.exists IS TRUE"
#         elif str(self.server_folder_path).startswith("/groups"):
#             filters = f"WHERE {table_name_shortcut}.exists_prfs IS TRUE"
#         elif str(self.server_folder_path).startswith("/aws") or str(self.server_folder_path).startswith(
#             "/workspace/CellObservatoryData"
#         ):
#             filters = f"WHERE {table_name_shortcut}.exists_aws IS TRUE"
#         elif str(self.server_folder_path).startswith("/lustre"):
#             filters = f"WHERE {table_name_shortcut}.exists_oak IS TRUE"
#         else:
#             raise ValueError(f"Unknown server_folder_path: {self.server_folder_path}")

#         if roi_list is not None or tile_list is not None:
#             filters += self._choose_filter(rois=roi_list, tiles=tile_list, table_name=table_name_shortcut).replace(
#                 "WHERE", " AND "
#             )
#         elif max_rois is not None or max_tiles is not None:
#             filters += self._limit_filter(
#                 max_rois=max_rois, max_tiles=max_tiles, table_name=table_name_shortcut
#             ).replace("WHERE", " AND ")

#         if hpf_list is not None:
#             filters += self._age_filter(hpfs=hpf_list, table_name=table_name_shortcut).replace("WHERE", " AND ")

#         if self.verbose:
#             print(f"Using filters: {filters}")
#         return filters

#     def _age_filter(self, hpfs: Iterable[int], table_name: str = "ptv") -> str:
#         assert hpfs is not None, "hpfs must be provided"

#         hpfs = tuple(hpfs) if len(hpfs) > 1 else f"({hpfs[0]})"
#         return f"WHERE {table_name}.hpf IN {hpfs}"

#     def _limit_filter(
#         self,
#         max_rois: Optional[int] = None,
#         max_tiles: Optional[int] = None,
#         table_name: str = "ptv",
#         idx_col: str = "id",
#     ) -> str:
#         assert max_rois is not None or max_tiles is not None, "At least one of max_rois or max_tiles must be provided"

#         if max_rois is not None:
#             unique_rois = self.get_random_rois(max_rois)
#             if isinstance(unique_rois, Iterable):
#                 filters = f"WHERE {table_name}.{idx_col} IN {tuple(unique_rois)}"
#             else:
#                 filters = f"WHERE {table_name}.{idx_col} IN ('{unique_rois}')"
#         else:
#             if max_tiles > 1:
#                 unique_rois, unique_tiles = zip(*self.get_random_tiles(max_tiles))
#             else:
#                 unique_rois, unique_tiles = self.get_random_tiles(max_tiles)

#             if isinstance(unique_tiles, Iterable) and isinstance(unique_rois, Iterable):
#                 filters = (
#                     f"WHERE {table_name}.{idx_col} IN {tuple(unique_rois)} "
#                     f"AND {table_name}.tile_name IN {tuple(unique_tiles)}"
#                 )
#             else:
#                 filters = (
#                     f"WHERE {table_name}.{idx_col} IN ('{unique_rois}') "
#                     f"AND {table_name}.tile_name IN ('{unique_tiles}')"
#                 )
#         return filters

#     def _choose_filter(
#         self,
#         rois: Optional[Iterable[int | str]] = None,
#         tiles: Optional[Iterable[str]] = None,
#         table_name: str = "ptv",
#         idx_col: str = "id",
#     ) -> str:

#         def _sql_in_list(values):
#             out = []
#             for v in values:
#                 if isinstance(v, (int, float)) or (isinstance(v, str) and v.isnumeric()):
#                     out.append(str(v))
#                 else:
#                     out.append("'" + str(v).replace("'", "''") + "'")
#             return "(" + ",".join(out) + ")"

#         assert rois is not None or tiles is not None, "At least one of rois or tiles must be provided"

#         clauses = []
#         if rois is not None:
#             rois_list = list(rois)
#             clauses.append(f"{table_name}.{idx_col} IN {_sql_in_list(rois_list)}")

#         if tiles is not None:
#             tiles_list = list(tiles)
#             clauses.append(f"{table_name}.tile_name IN {_sql_in_list(tiles_list)}")

#         return "WHERE " + " AND ".join(clauses)

#     def count_rows(self, table_name: str) -> int:
#         return self.execute_query(f"SELECT COUNT(*) FROM {table_name};").iloc[0, 0]

#     def _predict(self, batch_tensor: torch.Tensor, data_sample: dict) -> torch.Tensor:
#         if self.task == "detection":
#             if self.decoder_head_type == "plaindetr":
#                 preds = self.model.predict(data_sample)
#             else:
#                 raise NotImplementedError(
#                     f"Decoder head type {self.decoder_head_type} not supported for detection sliding window inference."
#                 )

#         elif self.task == "instance_segmentation":
#             if self.decoder_head_type == "maskdino":
#                 preds = self.model.predict(data_sample)
#             else:
#                 raise NotImplementedError(
#                     f"Decoder head type {self.decoder_head_type} not supported for instance segmentation sliding window inference."
#                 )

#         elif self.task == "semantic_segmentation":
#             raise NotImplementedError("Semantic segmentation decoder head not yet supported for sliding window inference.")            

#         elif self.task == "dense_prediction":
#             if self.task not in {"upsample_space", "upsample_time", "upsample_space_time", "channel_split"}:
#                 raise NotImplementedError(
#                     f"Task {self.task} not implemented for {self.decoder_head_type} sliding window inference."
#                 )
#             pred_hypercubes = self.model.predict(data_sample)
#             preds = {self.main_output_name: pred_hypercubes}

#         elif self.task == "pretrain":
#             pred_hypercubes = self.model.predict(data_sample)
#             preds = {self.main_output_name: pred_hypercubes}

#         elif self.task == "feature_extractor":
#             # NOTE: we expect patchified prediction tensors here
#             pred_hypercubes = self.model.forward_features(batch_tensor, masks=None)
            
#             if self.input_format == "ZYXC":
#                 B, N, C = pred_hypercubes.shape

#                 # FIXME: this will break if the backbone does further downsampling
#                 num = int(np.prod(self.token_shape))
#                 # CLS token
#                 if N == num + 1:
#                     pred_hypercubes = pred_hypercubes[:, 1:, :]
#                     N -= 1
#                 if N != num:
#                     raise RuntimeError(f"forward_features returned N={N}, expected {num} (token_shape={self.token_shape})")

#                 pred_hypercubes = pred_hypercubes.view(B, *self.token_shape, C)
#                 pred_hypercubes = pred_hypercubes.unsqueeze(1)
#             else:
#                 raise ValueError("Feature extractor only supports ZYXC input format for now.")

#             preds = {self.main_output_name: pred_hypercubes}

#         else:
#             raise NotImplementedError(
#                 f"Decoder head type {self.decoder_head_type} not supported for sliding window inference."
#             )

#         return preds

#     def _build_state(self):
#         self._tile_state, self._row_by_key, self._name_by_key = {}, {}, {}
#         ws, rank = get_world_size(), process_rank()
#         self._key_by_rank = {}

#         for _, row in self.prediction_df.iterrows():
#             roi = int(row["id"])
#             name = str(row["tile_name"])
#             owner = stable_key_owner(roi, name, ws)
#             self._key_by_rank.setdefault(owner, []).append((roi, name))

#         for roi, name in self._key_by_rank.get(rank, []):
#             row = self.prediction_df[
#                 (self.prediction_df["id"] == roi) & (self.prediction_df["tile_name"] == name)
#             ].iloc[0]

#             vol_T = int(row["time_size"])
#             vol_Z = int(row["z_end"] - row["z_start"])
#             vol_Y = int(row["y_end"] - row["y_start"])
#             vol_X = int(row["x_end"] - row["x_start"])
#             out_spatial_shape = (vol_T, vol_Z, vol_Y, vol_X)

#             tile_rows = self.database[
#                 (self.database["prepared_id"].astype(int) == roi) & (self.database["tile_name"].astype(str) == name)
#             ]
#             n_cubes = int(tile_rows.shape[0])

#             z_size = int(tile_rows["z_size"].iloc[0])
#             y_size = int(tile_rows["y_size"].iloc[0])
#             x_size = int(tile_rows["x_size"].iloc[0])

#             t_per_cube = (
#                 int(tile_rows["time_size"].iloc[0])
#                 if (self.input_format == "TZYXC" and "time_size" in tile_rows.columns)
#                 else 1
#             )

#             voxels_per_cube = t_per_cube * z_size * y_size * x_size
#             tile_volume = n_cubes * voxels_per_cube

#             key = (roi, tile_hash(name))
#             st = {
#                 "row": row,
#             }
#             for dt in self.data_types:
#                 if dt == self.main_output_name:
#                     C = self.num_output_channels

#                     if self.task == "feature_extractor":
#                         # for feature visualization, output shape is feature grid
#                         # of tokens which are spatially smaller than the input tensor                        
#                         Tt = ceil_div(vol_T, self.temporal_patch_size)
#                         Zt = ceil_div(vol_Z, self.axial_patch_size)
#                         Yt = ceil_div(vol_Y, self.lateral_patch_size)
#                         Xt = ceil_div(vol_X, self.lateral_patch_size)
#                         out_shape = (Tt, Zt, Yt, Xt, C)
#                         tile_volume_dt = Tt * Zt * Yt * Xt

#                     else:
#                         out_shape = (*out_spatial_shape, C)
#                         tile_volume_dt = tile_volume

#                 else:
#                     meta = self.auxiliary_outputs.get(dt)
#                     C = meta.get("num_output_channels")
#                     assert C is not None, f"num_output_channels must be specified for auxiliary output {dt}"
#                     tile_volume_dt = tile_volume
#                     out_shape = (*out_spatial_shape, C)

#                 st[f"shape_{dt}"] = out_shape
#                 st[f"pred_{dt}"] = None
#                 st[f"cnt_{dt}"] = None
#                 st[f"remaining_{dt}"] = tile_volume_dt
#                 st[f"done_{dt}"] = False

#             self._tile_state[key] = st
#             self._row_by_key[key] = row
#             self._name_by_key[key] = name

#             ray.logger.info(f"Rank {rank} will process tile {name} (ROI {roi})")

#         os.makedirs(self.inference_save_dir, exist_ok=True)

#     def _build_pred_buckets(self, pred_hypercubes: torch.Tensor, meta: dict, with_token_grid: bool):
#         ws = get_world_size()
#         pred_buckets = {r: [] for r in range(ws)}
#         for b in range(pred_hypercubes.size(0)):
#             roi = int(meta["prepared_id"][b])
#             tile_nm = str(meta["tile_name"][b])
#             owner_rank = stable_key_owner(roi, tile_nm, ws)

#             t0 = int(meta["time_start"][b])
#             T = int(meta["time_size"][b])
#             t1 = t0 + T
#             z0 = int(meta["z_start"][b])
#             sz = int(meta["z_size"][b])
#             z1 = z0 + sz
#             y0 = int(meta["y_start"][b])
#             sy = int(meta["y_size"][b])
#             y1 = y0 + sy
#             x0 = int(meta["x_start"][b])
#             sx = int(meta["x_size"][b])
#             x1 = x0 + sx

#             patch = pred_hypercubes[b]
#             if self.input_format == "ZYXC" and patch.ndim == 4:
#                 patch = patch.unsqueeze(0)

#             if with_token_grid:
#                 st, sz, sy, sx = self.temporal_patch_size, self.axial_patch_size, self.lateral_patch_size, self.lateral_patch_size
#                 t0f, z0f, y0f, x0f = t0 // st, z0 // sz, y0 // sy, x0 // sx
#                 Tf, Zf, Yf, Xf = patch.shape[:4]  # patch is [Tt, Zt, Yt, Xt, C]
#                 coords = torch.tensor([t0f, t0f+Tf, z0f, z0f+Zf, y0f, y0f+Yf, x0f, x0f+Xf],
#                                     device=patch.device, dtype=torch.int32)
#             else:
#                 coords = torch.tensor([t0, t1, z0, z1, y0, y1, x0, x1], device=patch.device, dtype=torch.int32)

#             pred_buckets[owner_rank].append(
#                 (
#                     torch.tensor([roi, tile_hash(tile_nm)], device=patch.device, dtype=torch.long),
#                     coords,
#                     patch,
#                 )
#             )

#         return pred_buckets

#     def _pack_for_alltoall(
#         self,
#         chunks,  # List[(dst_rank, tensor)]
#         world_size: int,
#         device: torch.device,
#         tail_shape: tuple,
#         dtype: torch.dtype,
#     ):
#         outs, splits = [], []
#         for dst in range(world_size):
#             found = next((x for x in chunks if x[0] == dst), None)
#             if found is None:
#                 outs.append(torch.empty((0,) + tail_shape, device=device, dtype=dtype))
#                 splits.append(0)
#             else:
#                 outs.append(found[1])
#                 splits.append(found[1].size(0))
#         send = torch.cat(outs, dim=0) if outs else torch.empty((0,) + tail_shape, device=device, dtype=dtype)
#         return send, splits

#     # TODO: this could be generalized to handle more types of payloads
#     def _alltoall(self, buckets, metadata: dict, out_channels: int):
#         ws, rk = get_world_size(), process_rank()
#         device = torch.device("cuda", torch.cuda.current_device())

#         send_counts = torch.zeros(ws, dtype=torch.int32, device=device)
#         keys_bucket, coords_bucket, pred_hypercubes_bucket = [], [], []

#         for dst in range(ws):
#             # bucket: {rank: [(key, coord, patch), ...], ...}
#             bucket = buckets[dst]
#             if not bucket:
#                 continue
#             # keys: [N,2]
#             pred_keys = torch.stack([t[0] for t in bucket], dim=0)
#             # coords: [N,8]
#             pred_coords = torch.stack([t[1] for t in bucket], dim=0)
#             # patches: [N,T,S,S,S,C]
#             pred_hypercube = torch.stack([t[2] for t in bucket], dim=0)
#             keys_bucket.append((dst, pred_keys))
#             coords_bucket.append((dst, pred_coords))
#             pred_hypercubes_bucket.append((dst, pred_hypercube))
#             send_counts[dst] = pred_keys.size(0)

#         keys_send, keys_splits = self._pack_for_alltoall(
#             keys_bucket, world_size=ws, device=device, tail_shape=(2,), dtype=torch.long
#         )
#         coords_send, coords_splits = self._pack_for_alltoall(
#             coords_bucket, world_size=ws, device=device, tail_shape=(8,), dtype=torch.int32
#         )

#         # NOTE: assumes uniform hypercube shape with specific format
#         if pred_hypercubes_bucket:
#             tail = pred_hypercubes_bucket[0][1].shape[1:]  # (T,S,S,S,C)
#         else:
#             T = int(metadata["time_size"][0]) if "time_size" in metadata else 1
#             Sz = int(metadata["z_size"][0])
#             Sy = int(metadata["y_size"][0])
#             Sx = int(metadata["x_size"][0])
#             tail = (T, Sz, Sy, Sx, out_channels)

#         pred_hypercubes_send, pred_hypercubes_splits = self._pack_for_alltoall(
#             pred_hypercubes_bucket, world_size=ws, device=device, tail_shape=tail, dtype=self.dtype
#         )

#         send_counts_cuda = send_counts.to(device)
#         recv_counts_cuda = torch.empty_like(send_counts_cuda)
#         dist.all_to_all_single(output=recv_counts_cuda, input=send_counts_cuda)
#         recv_counts = recv_counts_cuda.cpu().tolist()
#         total_recv = sum(recv_counts)

#         if keys_send is None:
#             keys_recv = torch.empty((0, 2), device=device, dtype=torch.long)
#             coords_recv = torch.empty((0, 8), device=device, dtype=torch.int32)
#             pred_hypercubes_recv = (
#                 torch.empty((0,) + tail, device=device, dtype=pred_hypercubes_bucket[0][1].dtype) if tail else None
#             )
#         else:
#             keys_recv = torch.empty((total_recv, 2), device=device, dtype=keys_send.dtype)
#             coords_recv = torch.empty((total_recv, 8), device=device, dtype=coords_send.dtype)
#             pred_hypercubes_recv = torch.empty((total_recv,) + tail, device=device, dtype=pred_hypercubes_send.dtype)

#             dist.all_to_all_single(
#                 output=keys_recv, input=keys_send, output_split_sizes=recv_counts, input_split_sizes=keys_splits
#             )
#             dist.all_to_all_single(
#                 output=coords_recv, input=coords_send, output_split_sizes=recv_counts, input_split_sizes=coords_splits
#             )
#             dist.all_to_all_single(
#                 output=pred_hypercubes_recv,
#                 input=pred_hypercubes_send,
#                 output_split_sizes=recv_counts,
#                 input_split_sizes=pred_hypercubes_splits,
#             )

#         return keys_recv, coords_recv, pred_hypercubes_recv

#     def _apply_recv(self, keys, coords, pred_hypercubes, data_type: Optional[str] = None):
#         N = keys.size(0)
#         if N == 0:
#             return set()

#         if data_type is None:
#             data_type = self.main_output_name

#         done_keys = set()
#         pred_hypercubes, keys, coords = pred_hypercubes.to("cpu"), keys.to("cpu"), coords.to("cpu")
#         for i in range(N):
#             roi_id = int(keys[i, 0].item())
#             tile_h = int(keys[i, 1].item())
#             key = (roi_id, tile_h)

#             t0, t1, z0, z1, y0, y1, x0, x1 = coords[i].tolist()
#             pred_hypercube = pred_hypercubes[i]

#             pred_t, cnt_t = self._get_or_init_buffers(key, data_type=data_type)

#             pred_view = pred_t[t0:t1, z0:z1, y0:y1, x0:x1, :]
#             cnt_view = cnt_t[t0:t1, z0:z1, y0:y1, x0:x1, 0]

#             T2, Z2, Y2, X2, C2 = pred_view.shape
#             T, Z, Y, X, C = pred_hypercube.shape

#             # if patch is larger (because of padding), crop it
#             if (T != T2) or (Z != Z2) or (Y != Y2) or (X != X2):
#                 # we expect the patch to be >= view in each spatial dim
#                 if T < T2 or Z < Z2 or Y < Y2 or X < X2:
#                     raise RuntimeError(
#                         f"pred_hypercube smaller than pred_view: "
#                         f"patch {pred_hypercube.shape}, view {pred_view.shape}"
#                     )

#                 pred_hypercube = pred_hypercube[:T2, :Z2, :Y2, :X2, :]

#             zeros_before = (cnt_view == 0).sum()

#             if pred_hypercube.dtype != pred_view.dtype:
#                 pred_hypercube = pred_hypercube.to(pred_view.dtype)

#             pred_view.add_(pred_hypercube)
#             cnt_view.add_(1.0)

#             zeros_after = (cnt_view == 0).sum()
#             filled = int((zeros_before - zeros_after).item())
#             self._tile_state[key][f"remaining_{data_type}"] -= filled

#             # print(f"Tile state: {self._tile_state[key]["remaining_" + data_type]} remaining voxels for data type {data_type}")

#             if all(self._tile_state[key][f"remaining_{dt}"] <= 0 for dt in self.data_types):
#                 # TODO: this should most likely happen on a separate Actor
#                 #       to avoid blocking the main inference loop for saving files
#                 if self._finish_if_done(key):
#                     done_keys.add(key)

#         return done_keys

#     def _finish_if_done(self, key, force: bool = False):
#         st = self._tile_state[key]
#         if any(st[f"done_{dt}"] or st[f"pred_{dt}"] is None for dt in self.data_types):
#             return False
#         if not force and any(st[f"remaining_{dt}"] != 0 for dt in self.data_types):
#             return False

#         row = self._row_by_key[key]
#         name = self._name_by_key[key]
#         try:
#             base = str(row["output_folder"])
#         except KeyError:
#             base = f"inference_roi{row.get('id', 'unknown')}"

#         base_sample_name = base.replace("/", "_") + "_" + name
#         base_sample_name = base_sample_name.replace(".zarr", "").replace(".tiff", "")

#         preds_dict = {}
#         for dt in self.data_types:
#             p = st[f"pred_{dt}"]
#             c = st[f"cnt_{dt}"]

#             c.clamp_min_(1.0)
#             p.div_(c)
#             preds = p

#             # Optional per-output activation from outputs_metadata
#             meta = self.outputs_metadata.get(dt) or self.auxiliary_outputs.get(dt) or {}
#             activation = meta.get("activation", None)
#             if activation is not None:
#                 preds = activation(preds)

#             if self.timepoint_list is not None:
#                 preds = preds[self.timepoint_list, ...]

#             preds_dict[dt] = preds

#         if self.task == "feature_extractor":
#             save_feature_visualizations(
#                 name=base_sample_name,
#                 predictions=preds_dict,
#                 save_dir=self.inference_save_dir,
#                 gt_key="data_tensor",
#                 feat_key=self.main_output_name,
#                 z_step_pdf=self.z_step_pdf,
#                 pmin=self.pmin,
#                 pmax=self.pmax,
#                 fit="per_t",
#                 sample_voxels=50_000,
#                 viz=self.feature_viz_type,
#                 stride_zyx=(self.axial_patch_size, self.lateral_patch_size, self.lateral_patch_size)
#             )
#             print(f"Finished saving feature visualizations for tile {name} (ROI {row.get('id', 'unknown')})")
#         else:
#             save_predictions(
#                 name=base_sample_name,
#                 predictions=preds_dict,
#                 save_dir=self.inference_save_dir,
#                 save_as_volume=self.save_as_volume,
#                 save_as_pdf=self.save_as_pdf,
#                 z_step_pdf=self.z_step_pdf,
#                 filetype=self.inference_save_format,
#                 zarr_chunk_shape=self.inference_zarr_chunk_shape,
#                 zarr_shard_shape=self.inference_zarr_shard_shape,
#                 pmin=self.pmin,
#                 pmax=self.pmax,
#             )

#             print(f"Finished saving predictions for tile {name} (ROI {row.get('id', 'unknown')})")
#             print(
#                 f"[finish_if_done] key={key}, base_sample_name={base_sample_name}, "
#                 f"save_dir={self.inference_save_dir}, vizualize_outputs={self.vizualize_outputs}, "
#                 f"save_as_volume={self.save_as_volume}"
#             )

#         for dt in self.data_types:
#             st[f"done_{dt}"] = True

#         return True

#     def _get_or_init_buffers(self, key, data_type: str):
#         st = self._tile_state[key]
#         if st[f"pred_{data_type}"] is None:
#             st[f"pred_{data_type}"] = torch.zeros(st[f"shape_{data_type}"], dtype=self.dtype, device="cpu")
#             st[f"cnt_{data_type}"] = torch.zeros((*st[f"shape_{data_type}"][:-1], 1), dtype=torch.float16, device="cpu")
#         return st[f"pred_{data_type}"], st[f"cnt_{data_type}"]

#     _PATH_TOKEN = re.compile(r"""
#         ([^.[]+)
#         (?:\[(\d+)\])?
#     """, re.X)

#     @staticmethod
#     def resolve_path(root: Any, path: str) -> Any:
#         """
#         Resolve 'path' against 'root'.
#         Supports dict keys, attributes, and list indices via '[i]' or bare '.i'.
#         Examples:
#         - "data_tensor"
#         - "metainfo.masks[0]"
#         - "metainfo.masks.0"   (treated like index 0)
#         """
#         cur = root
#         for part in path.split("."):
#             if part == "":
#                 continue

#             # allow bare numeric segments as list indices (".0")
#             if part.isdigit():
#                 cur = cur[int(part)]
#                 continue

#             # support "key" and "key[i]"
#             m = InferencerWorker._PATH_TOKEN.fullmatch(part)
#             if not m:
#                 raise KeyError(f"Bad path segment: {part!r} (full path: {path!r})")
#             key, idx = m.group(1), m.group(2)

#             # descend by key/attr
#             if isinstance(cur, dict):
#                 cur = cur[key]
#             else:
#                 cur = getattr(cur, key)

#             # optional index
#             if idx is not None:
#                 cur = cur[int(idx)]

#         return cur

#     @staticmethod
#     def _iter_aux_specs(auxiliary_outputs: Any) -> Iterable[Tuple[str, Dict[str, Any]]]:
#         """
#         Yields (name, spec) pairs from either:
#         - list of {name, path, ...}
#         - dict name -> {path, ...}
#         """
#         if auxiliary_outputs is None:
#             return []
#         if isinstance(auxiliary_outputs, dict):
#             return auxiliary_outputs.items()
#         # assume list-like
#         return ((spec["name"], spec) for spec in auxiliary_outputs)

#     def _preprocess(self, data_sample: dict) -> dict:
#         """
#         Materialize + normalize all outputs that will be saved
#         """
#         specs: Dict[str, Dict[str, Any]] = {}

#         for name, meta in self.outputs_metadata.items():
#             m = dict(meta) if meta is not None else {}
#             specs[name] = m

#         for aux_name, spec in self.auxiliary_outputs.items():
#             s = dict(spec) if spec is not None else {}
#             specs[aux_name] = s

#         for name, spec in specs.items():
#             path = spec.get("path", name)

#             try:
#                 x = self.resolve_path(data_sample, path)
#             except Exception:
#                 continue  # predictions etc not in data_sample yet

#             if not isinstance(x, torch.Tensor):
#                 raise TypeError(f"{name}: expected torch.Tensor at preprocessing, got {type(x)} (path={path!r})")

#             if bool(spec.get("patchified", False)):
#                 x = self.pe_unpatchify(x, out_channels=spec.get("num_output_channels"))

#             data_sample[name] = x

#         return data_sample

#     def predict(self, data_sample: dict):
#         data_sample = self._preprocess(data_sample)

#         X = data_sample["data_tensor"]
#         metadata = data_sample["metainfo"]

#         preds = self._predict(X, data_sample)

#         if self.aggregate_mode == "stitch_volume":
#             pred_hypercubes = preds[self.main_output_name]
#             if not isinstance(pred_hypercubes, torch.Tensor):
#                 raise TypeError(
#                     f"stitch_volume requires Tensor preds[{self.main_output_name!r}], got {type(pred_hypercubes)}"
#                 )

#             buckets = self._build_pred_buckets(
#                 pred_hypercubes,
#                 metadata,
#                 with_token_grid=(self.task == "feature_extractor"),
#             )
#             keys_recv, coords_recv, pred_hypercubes_recv = self._alltoall(
#                 buckets, metadata, out_channels=self.num_output_channels
#             )
#             self._apply_recv(keys_recv, coords_recv, pred_hypercubes_recv, data_type=self.main_output_name)

#             if self.save_auxiliary_outputs:
#                 for aux_name, spec in self._iter_aux_specs(self.auxiliary_outputs):
#                     out_ch = spec["num_output_channels"]
#                     aux_pred_hypercubes = data_sample[aux_name]
#                     aux_buckets = self._build_pred_buckets(aux_pred_hypercubes, metadata, with_token_grid=False)
#                     aux_keys_recv, aux_coords_recv, aux_pred_hypercubes_recv = self._alltoall(
#                         aux_buckets, metadata, out_channels=out_ch
#                     )
#                     self._apply_recv(aux_keys_recv, aux_coords_recv, aux_pred_hypercubes_recv, data_type=aux_name)

#         elif self.aggregate_mode == "save_local":
#             # targets may be absent in pure inference; normalize to per-record list[dict]
#             B = len(metadata["prepared_id"])
#             targets = metadata.get("targets", None)
#             if targets is None:
#                 targets = [{} for _ in range(B)]
#             elif isinstance(targets, dict):
#                 targets = [targets for _ in range(B)]
#             elif isinstance(targets, (list, tuple)) and len(targets) == 1 and isinstance(targets[0], (list, tuple)):
#                 # common pattern: targets wrapped once
#                 targets = targets[0]

#             self._save_local_records(
#                 data_sample=data_sample,
#                 preds=preds,
#                 # FIXME: generalize
#                 targets=targets,
#                 metadata=metadata
#             )

#         else:
#             raise ValueError(f"Unknown aggregate_mode: {self.aggregate_mode}")

#     def _save_local_records(self, 
#                             data_sample: dict,
#                             preds: Dict[str, Any], 
#                             targets: List[Dict[str, Any]],
#                             metadata: dict
#     ):
#         """
#         No comms, no stitching: save/plot per-cube (per-rank local) records.
#         """
#         # determine batch size from metainfo
#         if "prepared_id" not in metadata:
#             raise KeyError("metainfo must contain 'prepared_id' for save_local mode")
#         B = len(metadata["prepared_id"])

#         rank_dir = Path(self.inference_save_dir) / f"rank{self.rank:03d}"
#         os.makedirs(rank_dir, exist_ok=True)

#         regions: List[Dict[str, Any]] = []
#         identifiers: List[str] = []

#         for b in range(B):
#             roi = int(metadata["prepared_id"][b])
#             tile_nm = str(metadata["tile_name"][b])

#             t0 = int(metadata["time_start"][b])
#             T = int(metadata["time_size"][b])
#             t1 = t0 + T

#             z0 = int(metadata["z_start"][b])
#             sz = int(metadata["z_size"][b])
#             z1 = z0 + sz
#             y0 = int(metadata["y_start"][b])
#             sy = int(metadata["y_size"][b])
#             y1 = y0 + sy
#             x0 = int(metadata["x_start"][b])
#             sx = int(metadata["x_size"][b])
#             x1 = x0 + sx

#             region = dict(
#                 roi=roi,
#                 tile_name=tile_nm,
#                 coords=(t0, t1, z0, z1, y0, y1, x0, x1),
#                 coord_frame="voxel",
#             )
#             ident = (
#                 f"rank{self.rank:03d}_roi{roi}_{tile_nm}"
#                 f"_t{t0}-{t1}_z{z0}-{z1}_y{y0}-{y1}_x{x0}-{x1}"
#             )

#             regions.append(region)
#             identifiers.append(ident)

#         save_instance_predictions(
#             # save_instance_predictions expects BTZYXC format
#             images=data_sample["data_tensor"].unsqueeze(1),
#             save_dir=rank_dir,
#             identifiers=identifiers,
#             preds=preds,
#             targets=targets,
#             regions=regions,
#             pred_boxes_format=self.pred_boxes_format,
#             gt_boxes_format=self.gt_boxes_format,
#             scale_gt_boxes=self.scale_gt_boxes,
#             input_format=self.input_format,
#             ortho=True,
#         )

#     def finalize(self):
#         if self.stitch_volume:
#             for key in list(self._tile_state.keys()):
#                 self._finish_if_done(key, force=True)
#         barrier()




