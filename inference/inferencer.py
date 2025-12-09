import logging
import os
from pathlib import Path
from typing import Callable, Iterable, List, Literal, Optional, Tuple

import connectorx as cx
import numpy as np
import pandas as pd
import ray
import torch
from dotenv import load_dotenv
from torch import distributed as dist

from cell_observatory_platform.data.io import save_file
from cell_observatory_platform.inference.utils import save_predictions, stable_key_owner, tile_hash
from cell_observatory_platform.utils.context import barrier, get_world_size, process_rank

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class InferencerWorker:
    def __init__(
        self,
        task: str,
        outputs_metadata: dict,
        model: torch.nn.Module,
        database: pd.DataFrame,
        use_cached_hypercubes_dataframe: bool,
        hypercubes_dataframe_path: Path,
        timepoint_list: Optional[List[int]],
        save_as_pdf: bool,
        save_as_volume: bool,
        z_step_pdf: int,
        input_format: str,
        inference_mode: str,
        decoder_head_type: str,
        roi_tile_list: List[Tuple[int, str]],
        save_dir: Path | str,
        dtype: torch.dtype = torch.float16,
        protocol: Literal["binary", "csv", "cursor"] = "binary",
        dotenv_path: Optional[Path] = Path(__file__).parent.parent.parent / ".env",
        dbname: Literal["staging", "prod"] = "prod",
        server_folder_path: Optional[Path] = None,
        verbose: bool = False,
        max_hypercubes: Optional[int] = None,
        max_partitions: int = 10,
        save_format: Literal["tiff", "zarr"] = "tiff",
        zarr_chunk_shape: Optional[Tuple[int, ...]] = None,
        zarr_shard_shape: Optional[Tuple[int, ...]] = None,
        auxiliary_outputs: Optional[List[Tuple[str, dict]]] = None,
    ):
        self.database = database
        self.hypercubes_dataframe_path = Path(hypercubes_dataframe_path)
        self.use_cached_hypercubes_dataframe = use_cached_hypercubes_dataframe
        # TODO: consider alternative methods for only performing inference
        #       on a subset of timepoints without storing a prediction for all timepoints
        self.timepoint_list = timepoint_list

        self.dtype = dtype
        self.input_format = input_format

        # roi_tile_list is a list of (roi_id, tile_name) tuples
        # to restrict inference to
        self.roi_tile_list = roi_tile_list
        self.roi_list = list(set([x[0] for x in roi_tile_list]))
        self.tile_list = list(set([x[1] for x in roi_tile_list]))

        self.model = model

        self.task = task
        self.inference_mode = inference_mode

        self.z_step_pdf = z_step_pdf
        self.save_as_pdf = save_as_pdf
        self.save_as_volume = save_as_volume

        self.decoder_head_type = decoder_head_type

        self.inference_save_dir = save_dir
        self.inference_save_format = save_format
        self.inference_zarr_chunk_shape = zarr_chunk_shape
        self.inference_zarr_shard_shape = zarr_shard_shape

        self.dbname = dbname
        self.verbose = verbose
        self.protocol = protocol
        self.dotenv_path = dotenv_path
        self.server_folder_path = server_folder_path
        self._database_url = self._load_uri()

        self.max_hypercubes = max_hypercubes
        self.max_partitions = max_partitions

        assert outputs_metadata is not None, "outputs_metadata must be provided"
        # DictConfig -> plain dict[str, dict]
        self.outputs_metadata = {str(name): dict(meta) for name, meta in outputs_metadata.items()}

        if auxiliary_outputs is not None:
            self.auxiliary_outputs = {str(name): dict(meta) for name, meta in auxiliary_outputs.items()}
        else:
            self.auxiliary_outputs = {}

        self.save_auxiliary_outputs = bool(self.auxiliary_outputs)

        # main prediction output name; assume first key in outputs_metadata
        self.main_output_name = next(iter(self.outputs_metadata.keys()))
        self.num_output_channels = self.outputs_metadata[self.main_output_name].get("num_output_channels")
        assert self.num_output_channels is not None, "num_output_channels must be specified for main output"

        # all data types we will aggregate/save:
        #   - main output (e.g. 'predictions')
        #   - plus each auxiliary output (e.g. 'data_tensor')
        self.data_types = [self.main_output_name, *self.auxiliary_outputs.keys()]

        self.prediction_df = self._get_data_tiles_metadata()
        self._build_state()

        ray.logger.info(f"Inference Database: {self.prediction_df}")
        ray.logger.info(f"Data types to save: {self.data_types}")
        ray.logger.info(f"Auxiliary outputs: {self.auxiliary_outputs}")
        ray.logger.info(f"Main output metadata: {self.outputs_metadata}")

    def _get_data_tiles_metadata(self) -> pd.DataFrame:
        roi_csv = self.hypercubes_dataframe_path.with_name(f"{self.hypercubes_dataframe_path.stem}_rois.csv")
        if (not self.use_cached_hypercubes_dataframe) or (not roi_csv.exists()):
            query = self._get_query(roi_list=self.roi_list)
            table = self.execute_query(query)
        else:
            if self.verbose:
                print(f"Loading hypercubes dataframe from cached file: {roi_csv}")
            table = pd.read_csv(roi_csv)

            # FIXME: maintain uniform naming accross databases
            #        to avoid this
            cols_rename = [
                "tile_z_start",
                "tile_y_start",
                "tile_x_start",
                "tile_z_end",
                "tile_y_end",
                "tile_x_end",
                "tile_channel_size",
                "tile_time_size",
            ]
            if set(cols_rename).issubset(table.columns):
                table = table.rename(
                    columns={
                        "prepared_id": "id",
                        "tile_z_end": "z_end",
                        "tile_y_end": "y_end",
                        "tile_x_end": "x_end",
                        "tile_z_start": "z_start",
                        "tile_y_start": "y_start",
                        "tile_x_start": "x_start",
                        "tile_time_size": "time_size",
                        "tile_channel_size": "channel_size",
                    }
                )

            table = table[table["id"].isin(self.roi_list)]

            print(f"Loaded Table: {table}")

        roi_ids = table["id"].tolist()
        tiles = self._get_tiles_from_rois(roi_ids)
        table = table.merge(tiles, left_on="id", right_on="prepared_id", how="left")

        allowed = pd.DataFrame(self.roi_tile_list, columns=["id", "tile_name"]).drop_duplicates()
        table = table.merge(allowed, on=["id", "tile_name"], how="inner")

        # table = table.drop_duplicates(subset=["tile_name", "output_folder"], keep="first")

        table["prediction"] = [None] * len(table)
        table["count"] = [None] * len(table)

        return table

    def _load_uri(self):
        if "SUPABASE_STAGING_URI" not in os.environ or "SUPABASE_PROD_URI" not in os.environ:
            assert Path(self.dotenv_path).exists(), f"{self.dotenv_path} was not found"
            if self.verbose:
                print(f"Loading additional environment variables from {self.dotenv_path}")
            load_dotenv(self.dotenv_path, verbose=True)

        if self.dbname == "staging":
            uri = os.environ.get("SUPABASE_STAGING_URI")
        elif self.dbname == "prod":
            uri = os.environ.get("SUPABASE_PROD_URI")
        else:
            raise ValueError(f"Unknown database name: {self.dbname}")

        assert uri is not None, "SUPABASE_URI_* environment variable not set"
        return uri

    def execute_query(self, query: str | List[str]) -> pd.DataFrame:
        try:
            # avoid the costly COUNT query for pandas by using arrow as an intermediate step
            # https://sfu-db.github.io/connector-x/freq_questions.html
            result = cx.read_sql(
                conn=self._database_url,
                query=query,
                protocol=self.protocol,
                return_type="arrow",
            )
            df = result.to_pandas(split_blocks=False, date_as_object=False)
            return df
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            raise

    def _get_tiles_from_rois(
        self,
        roi_ids: list,
        table_name_shortcut: str = "hc",
        tile_list: Optional[list] = None,
        table_name: str = "prepared_tiles",
        idx_col: str = "prepared_id",
    ) -> pd.DataFrame:
        query = f"""
            SELECT
                *
            FROM {table_name}
            WHERE {idx_col} IN ({', '.join(map(str, roi_ids))})
        """
        table = self.execute_query(query)
        if tile_list is not None:
            table = table[table["tile_name"].isin(tile_list)]
        return table

    def _get_query(
        self,
        roi_list: list,
        tile_list: Optional[list] = None,
        column_names: list = [
            "id",
            "y_start",
            "x_start",
            "z_start",
            "y_end",
            "x_end",
            "z_end",
            "channel_size",
            "time_size",
            "output_folder",
        ],
        table_name: str = "prepared",
        table_name_shortcut: str = "hc",
        idx_col: str = "id",
    ) -> str:
        filters = self._filters_to_string(
            table_name_shortcut=table_name_shortcut,
            roi_list=roi_list,
            tile_list=tile_list,
        )
        max_rows = self.count_rows(table_name=table_name)

        if self.max_hypercubes is None:
            max_hypercubes = max_rows
        else:
            max_hypercubes = self.max_hypercubes

        if max_hypercubes > max_rows:
            max_hypercubes = max_rows

        if max_hypercubes > 1000:
            # select max number of partitions that divides the number of rows in each partition evenly
            partition_num = (
                max([i for i in range(1, self.max_partitions + 1) if max_hypercubes % i == 0])
                if max_hypercubes is not None
                else 1
            )
            print(f"Using {partition_num} partitions to query.")
        else:
            partition_num = 1

        rows_per_partition = max_hypercubes // partition_num
        queries = [
            f"""
                SELECT
                    {', '.join([f'{table_name_shortcut}.{col}' for col in column_names])}
                FROM {table_name} {table_name_shortcut}
                {filters}
                ORDER BY {idx_col} DESC
                LIMIT {rows_per_partition}
                OFFSET {rows_per_partition * i}
            """
            for i in range(partition_num)
        ]

        return queries

    def _filters_to_string(
        self,
        table_name_shortcut: str = "hc",
        max_rois: Optional[int] = None,
        max_tiles: Optional[int] = None,
        hpf_list: Optional[Iterable[int]] = None,
        roi_list: Optional[Iterable[int]] = None,
        tile_list: Optional[Iterable[str]] = None,
    ) -> str:

        if self.server_folder_path is None or str(self.server_folder_path).startswith("/clusterfs"):
            filters = f"WHERE {table_name_shortcut}.exists IS TRUE"
        elif str(self.server_folder_path).startswith("/groups"):
            filters = f"WHERE {table_name_shortcut}.exists_prfs IS TRUE"
        elif str(self.server_folder_path).startswith("/aws") or str(self.server_folder_path).startswith(
            "/workspace/CellObservatoryData"
        ):
            filters = f"WHERE {table_name_shortcut}.exists_aws IS TRUE"
        elif str(self.server_folder_path).startswith("/lustre"):
            filters = f"WHERE {table_name_shortcut}.exists_oak IS TRUE"
        else:
            raise ValueError(f"Unknown server_folder_path: {self.server_folder_path}")

        if roi_list is not None or tile_list is not None:
            filters += self._choose_filter(rois=roi_list, tiles=tile_list, table_name=table_name_shortcut).replace(
                "WHERE", " AND "
            )
        elif max_rois is not None or max_tiles is not None:
            filters += self._limit_filter(
                max_rois=max_rois, max_tiles=max_tiles, table_name=table_name_shortcut
            ).replace("WHERE", " AND ")

        if hpf_list is not None:
            filters += self._age_filter(hpfs=hpf_list, table_name=table_name_shortcut).replace("WHERE", " AND ")

        if self.verbose:
            print(f"Using filters: {filters}")
        return filters

    def _age_filter(self, hpfs: Iterable[int], table_name: str = "ptv") -> str:
        assert hpfs is not None, "hpfs must be provided"

        hpfs = tuple(hpfs) if len(hpfs) > 1 else f"({hpfs[0]})"
        return f"WHERE {table_name}.hpf IN {hpfs}"

    def _limit_filter(
        self,
        max_rois: Optional[int] = None,
        max_tiles: Optional[int] = None,
        table_name: str = "ptv",
        idx_col: str = "id",
    ) -> str:
        assert max_rois is not None or max_tiles is not None, "At least one of max_rois or max_tiles must be provided"

        if max_rois is not None:
            unique_rois = self.get_random_rois(max_rois)
            if isinstance(unique_rois, Iterable):
                filters = f"WHERE {table_name}.{idx_col} IN {tuple(unique_rois)}"
            else:
                filters = f"WHERE {table_name}.{idx_col} IN ('{unique_rois}')"
        else:
            if max_tiles > 1:
                unique_rois, unique_tiles = zip(*self.get_random_tiles(max_tiles))
            else:
                unique_rois, unique_tiles = self.get_random_tiles(max_tiles)

            if isinstance(unique_tiles, Iterable) and isinstance(unique_rois, Iterable):
                filters = (
                    f"WHERE {table_name}.{idx_col} IN {tuple(unique_rois)} "
                    f"AND {table_name}.tile_name IN {tuple(unique_tiles)}"
                )
            else:
                filters = (
                    f"WHERE {table_name}.{idx_col} IN ('{unique_rois}') "
                    f"AND {table_name}.tile_name IN ('{unique_tiles}')"
                )
        return filters

    def _choose_filter(
        self,
        rois: Optional[Iterable[int | str]] = None,
        tiles: Optional[Iterable[str]] = None,
        table_name: str = "ptv",
        idx_col: str = "id",
    ) -> str:

        def _sql_in_list(values):
            out = []
            for v in values:
                if isinstance(v, (int, float)) or (isinstance(v, str) and v.isnumeric()):
                    out.append(str(v))
                else:
                    out.append("'" + str(v).replace("'", "''") + "'")
            return "(" + ",".join(out) + ")"

        assert rois is not None or tiles is not None, "At least one of rois or tiles must be provided"

        clauses = []
        if rois is not None:
            rois_list = list(rois)
            clauses.append(f"{table_name}.{idx_col} IN {_sql_in_list(rois_list)}")

        if tiles is not None:
            tiles_list = list(tiles)
            clauses.append(f"{table_name}.tile_name IN {_sql_in_list(tiles_list)}")

        return "WHERE " + " AND ".join(clauses)

    def count_rows(self, table_name: str) -> int:
        return self.execute_query(f"SELECT COUNT(*) FROM {table_name};").iloc[0, 0]

    def _predict(self, batch_tensor: torch.Tensor, data_sample: dict) -> torch.Tensor:
        if self.decoder_head_type == "mask2former":
            if self.task != "semantic_segmentation":
                raise NotImplementedError(f"Task {self.task} not implemented for Mask2Former sliding window inference.")
            crop_out = self.model.predict(batch_tensor, rescale_to=batch_tensor.shape[2:])
            mask_pred = crop_out["pred_masks"].sigmoid()
            mask_cls = torch.softmax(crop_out["pred_logits"], dim=-1)[..., :-1]
            pred_hypercubes = torch.einsum("bqc,bq...->bc...", mask_cls, mask_pred)

        elif self.decoder_head_type in {"vit", "linear"}:
            if self.task not in {"upsample_space", "upsample_time", "upsample_space_time", "channel_split"}:
                raise NotImplementedError(
                    f"Task {self.task} not implemented for {self.decoder_head_type} sliding window inference."
                )
            pred_hypercubes = self.model.predict(data_sample)

        elif self.decoder_head_type == "maskdino":
            raise NotImplementedError("MaskDINO decoder head not yet supported for sliding window inference.")

        elif self.decoder_head_type == "dpt":
            raise NotImplementedError("Dense Prediction Head not yet supported for sliding window inference.")

        elif self.decoder_head_type == "pretrain":
            pred_hypercubes = self.model.predict(data_sample)

        else:
            raise NotImplementedError(
                f"Decoder head type {self.decoder_head_type} not supported for sliding window inference."
            )

        return pred_hypercubes

    def _build_state(self):
        self._tile_state, self._row_by_key, self._name_by_key = {}, {}, {}
        ws, rank = get_world_size(), process_rank()
        self._key_by_rank = {}

        for _, row in self.prediction_df.iterrows():
            roi = int(row["id"])
            name = str(row["tile_name"])
            owner = stable_key_owner(roi, name, ws)
            self._key_by_rank.setdefault(owner, []).append((roi, name))

        for roi, name in self._key_by_rank.get(rank, []):
            row = self.prediction_df[
                (self.prediction_df["id"] == roi) & (self.prediction_df["tile_name"] == name)
            ].iloc[0]

            vol_T = int(row["time_size"])
            vol_Z = int(row["z_end"] - row["z_start"])
            vol_Y = int(row["y_end"] - row["y_start"])
            vol_X = int(row["x_end"] - row["x_start"])
            out_spatial_shape = (vol_T, vol_Z, vol_Y, vol_X)

            tile_rows = self.database[
                (self.database["prepared_id"].astype(int) == roi) & (self.database["tile_name"].astype(str) == name)
            ]
            n_cubes = int(tile_rows.shape[0])

            z_size = int(tile_rows["z_size"].iloc[0])
            y_size = int(tile_rows["y_size"].iloc[0])
            x_size = int(tile_rows["x_size"].iloc[0])

            t_per_cube = (
                int(tile_rows["time_size"].iloc[0])
                if (self.input_format == "TZYXC" and "time_size" in tile_rows.columns)
                else 1
            )

            voxels_per_cube = t_per_cube * z_size * y_size * x_size
            tile_volume = n_cubes * voxels_per_cube

            key = (roi, tile_hash(name))
            st = {
                "row": row,
            }
            for dt in self.data_types:
                if dt == self.main_output_name:
                    C = self.num_output_channels
                else:
                    meta = self.auxiliary_outputs.get(dt)
                    C = meta.get("num_output_channels")
                    assert C is not None, f"num_output_channels must be specified for auxiliary output {dt}"

                st[f"shape_{dt}"] = (*out_spatial_shape, C)
                st[f"pred_{dt}"] = None
                st[f"cnt_{dt}"] = None
                st[f"remaining_{dt}"] = tile_volume
                st[f"done_{dt}"] = False

            self._tile_state[key] = st
            self._row_by_key[key] = row
            self._name_by_key[key] = name

            ray.logger.info(f"Rank {rank} will process tile {name} (ROI {roi})")

        os.makedirs(self.inference_save_dir, exist_ok=True)

    def _build_pred_buckets(self, pred_hypercubes: torch.Tensor, meta: dict):
        ws = get_world_size()
        pred_buckets = {r: [] for r in range(ws)}
        for b in range(pred_hypercubes.size(0)):
            roi = int(meta["prepared_id"][b])
            tile_nm = str(meta["tile_name"][b])
            owner_rank = stable_key_owner(roi, tile_nm, ws)

            t0 = int(meta["time_start"][b])
            T = int(meta["time_size"][b])
            t1 = t0 + T
            z0 = int(meta["z_start"][b])
            sz = int(meta["z_size"][b])
            z1 = z0 + sz
            y0 = int(meta["y_start"][b])
            sy = int(meta["y_size"][b])
            y1 = y0 + sy
            x0 = int(meta["x_start"][b])
            sx = int(meta["x_size"][b])
            x1 = x0 + sx

            patch = pred_hypercubes[b]
            if self.input_format == "ZYXC" and patch.ndim == 4:
                patch = patch.unsqueeze(0)
            pred_buckets[owner_rank].append(
                (
                    torch.tensor([roi, tile_hash(tile_nm)], device=patch.device, dtype=torch.long),
                    torch.tensor([t0, t1, z0, z1, y0, y1, x0, x1], device=patch.device, dtype=torch.int32),
                    patch,
                )
            )
        return pred_buckets

    def _pack_for_alltoall(
        self,
        chunks,  # List[(dst_rank, tensor)]
        world_size: int,
        device: torch.device,
        tail_shape: tuple,
        dtype: torch.dtype,
    ):
        outs, splits = [], []
        for dst in range(world_size):
            found = next((x for x in chunks if x[0] == dst), None)
            if found is None:
                outs.append(torch.empty((0,) + tail_shape, device=device, dtype=dtype))
                splits.append(0)
            else:
                outs.append(found[1])
                splits.append(found[1].size(0))
        send = torch.cat(outs, dim=0) if outs else torch.empty((0,) + tail_shape, device=device, dtype=dtype)
        return send, splits

    # TODO: this could be generalized to handle more types of payloads
    def _alltoall(self, buckets, metadata: dict, out_channels: int):
        ws, rk = get_world_size(), process_rank()
        device = torch.device("cuda", torch.cuda.current_device())

        send_counts = torch.zeros(ws, dtype=torch.int32, device=device)
        keys_bucket, coords_bucket, pred_hypercubes_bucket = [], [], []

        for dst in range(ws):
            # bucket: {rank: [(key, coord, patch), ...], ...}
            bucket = buckets[dst]
            if not bucket:
                continue
            # keys: [N,2]
            pred_keys = torch.stack([t[0] for t in bucket], dim=0)
            # coords: [N,8]
            pred_coords = torch.stack([t[1] for t in bucket], dim=0)
            # patches: [N,T,S,S,S,C]
            pred_hypercube = torch.stack([t[2] for t in bucket], dim=0)
            keys_bucket.append((dst, pred_keys))
            coords_bucket.append((dst, pred_coords))
            pred_hypercubes_bucket.append((dst, pred_hypercube))
            send_counts[dst] = pred_keys.size(0)

        keys_send, keys_splits = self._pack_for_alltoall(
            keys_bucket, world_size=ws, device=device, tail_shape=(2,), dtype=torch.long
        )
        coords_send, coords_splits = self._pack_for_alltoall(
            coords_bucket, world_size=ws, device=device, tail_shape=(8,), dtype=torch.int32
        )

        # NOTE: assumes uniform hypercube shape with specific format
        if pred_hypercubes_bucket:
            tail = pred_hypercubes_bucket[0][1].shape[1:]  # (T,S,S,S,C)
        else:
            T = int(metadata["time_size"][0]) if "time_size" in metadata else 1
            Sz = int(metadata["z_size"][0])
            Sy = int(metadata["y_size"][0])
            Sx = int(metadata["x_size"][0])
            tail = (T, Sz, Sy, Sx, out_channels)

        pred_hypercubes_send, pred_hypercubes_splits = self._pack_for_alltoall(
            pred_hypercubes_bucket, world_size=ws, device=device, tail_shape=tail, dtype=self.dtype
        )

        send_counts_cuda = send_counts.to(device)
        recv_counts_cuda = torch.empty_like(send_counts_cuda)
        dist.all_to_all_single(output=recv_counts_cuda, input=send_counts_cuda)
        recv_counts = recv_counts_cuda.cpu().tolist()
        total_recv = sum(recv_counts)

        if keys_send is None:
            keys_recv = torch.empty((0, 2), device=device, dtype=torch.long)
            coords_recv = torch.empty((0, 8), device=device, dtype=torch.int32)
            pred_hypercubes_recv = (
                torch.empty((0,) + tail, device=device, dtype=pred_hypercubes_bucket[0][1].dtype) if tail else None
            )
        else:
            keys_recv = torch.empty((total_recv, 2), device=device, dtype=keys_send.dtype)
            coords_recv = torch.empty((total_recv, 8), device=device, dtype=coords_send.dtype)
            pred_hypercubes_recv = torch.empty((total_recv,) + tail, device=device, dtype=pred_hypercubes_send.dtype)

            dist.all_to_all_single(
                output=keys_recv, input=keys_send, output_split_sizes=recv_counts, input_split_sizes=keys_splits
            )
            dist.all_to_all_single(
                output=coords_recv, input=coords_send, output_split_sizes=recv_counts, input_split_sizes=coords_splits
            )
            dist.all_to_all_single(
                output=pred_hypercubes_recv,
                input=pred_hypercubes_send,
                output_split_sizes=recv_counts,
                input_split_sizes=pred_hypercubes_splits,
            )

        return keys_recv, coords_recv, pred_hypercubes_recv

    def _apply_recv(self, keys, coords, pred_hypercubes, data_type: Optional[str] = None):
        N = keys.size(0)
        if N == 0:
            return set()

        if data_type is None:
            data_type = self.main_output_name

        done_keys = set()
        pred_hypercubes, keys, coords = pred_hypercubes.to("cpu"), keys.to("cpu"), coords.to("cpu")
        for i in range(N):
            roi_id = int(keys[i, 0].item())
            tile_h = int(keys[i, 1].item())
            key = (roi_id, tile_h)

            t0, t1, z0, z1, y0, y1, x0, x1 = coords[i].tolist()
            pred_hypercube = pred_hypercubes[i]

            pred_t, cnt_t = self._get_or_init_buffers(key, data_type=data_type)

            pred_view = pred_t[t0:t1, z0:z1, y0:y1, x0:x1, :]
            cnt_view = cnt_t[t0:t1, z0:z1, y0:y1, x0:x1, 0]

            T2, Z2, Y2, X2, C2 = pred_view.shape
            T, Z, Y, X, C = pred_hypercube.shape

            # if patch is larger (because of padding), crop it
            if (T != T2) or (Z != Z2) or (Y != Y2) or (X != X2):
                # we expect the patch to be >= view in each spatial dim
                if T < T2 or Z < Z2 or Y < Y2 or X < X2:
                    raise RuntimeError(
                        f"pred_hypercube smaller than pred_view: "
                        f"patch {pred_hypercube.shape}, view {pred_view.shape}"
                    )

                pred_hypercube = pred_hypercube[:T2, :Z2, :Y2, :X2, :]

            zeros_before = (cnt_view == 0).sum()

            pred_view.add_(pred_hypercube)
            cnt_view.add_(1)

            zeros_after = (cnt_view == 0).sum()
            filled = int((zeros_before - zeros_after).item())
            self._tile_state[key][f"remaining_{data_type}"] -= filled

            # print(f"Tile state: {self._tile_state[key]["remaining_" + data_type]} remaining voxels for data type {data_type}")

            if all(self._tile_state[key][f"remaining_{dt}"] <= 0 for dt in self.data_types):
                # TODO: this should most likely happen on a separate Actor
                #       to avoid blocking the main inference loop for saving files
                if self._finish_if_done(key):
                    done_keys.add(key)

        return done_keys

    def _finish_if_done(self, key, force: bool = False):
        st = self._tile_state[key]
        if any(st[f"done_{dt}"] or st[f"pred_{dt}"] is None for dt in self.data_types):
            return False
        if not force and any(st[f"remaining_{dt}"] != 0 for dt in self.data_types):
            return False

        row = self._row_by_key[key]
        name = self._name_by_key[key]
        try:
            base = str(row["output_folder"])
        except KeyError:
            base = f"inference_roi{row.get('id', 'unknown')}"

        base_sample_name = base.replace("/", "_") + "_" + name
        base_sample_name = base_sample_name.replace(".zarr", "").replace(".tiff", "")

        preds_dict = {}
        for dt in self.data_types:
            preds = st[f"pred_{dt}"] / st[f"cnt_{dt}"].clamp_min(1)

            # Optional per-output activation from outputs_metadata
            meta = self.outputs_metadata.get(dt, {})
            activation = meta.get("activation", None)
            if activation is not None:
                preds = activation(preds)

            if self.timepoint_list is not None:
                preds = preds[self.timepoint_list, ...]

            preds_dict[dt] = preds

        save_predictions(
            name=base_sample_name,
            predictions=preds_dict,
            save_dir=self.inference_save_dir,
            save_as_volume=self.save_as_volume,
            save_as_pdf=self.save_as_pdf,
            z_step_pdf=self.z_step_pdf,
            filetype=self.inference_save_format,
            zarr_chunk_shape=self.inference_zarr_chunk_shape,
            zarr_shard_shape=self.inference_zarr_shard_shape,
        )

        print(f"Finished saving predictions for tile {name} (ROI {row.get('id', 'unknown')})")
        print(
            f"[finish_if_done] key={key}, base_sample_name={base_sample_name}, "
            f"save_dir={self.inference_save_dir}, save_as_pdf={self.save_as_pdf}, "
            f"save_as_volume={self.save_as_volume}"
        )

        for dt in self.data_types:
            st[f"done_{dt}"] = True

        return True

    def _get_or_init_buffers(self, key, data_type: str):
        st = self._tile_state[key]
        if st[f"pred_{data_type}"] is None:
            st[f"pred_{data_type}"] = torch.zeros(st[f"shape_{data_type}"], dtype=self.dtype, device="cpu")
            st[f"cnt_{data_type}"] = torch.zeros((*st[f"shape_{data_type}"][:-1], 1), dtype=torch.int32, device="cpu")
        return st[f"pred_{data_type}"], st[f"cnt_{data_type}"]

    def predict(self, data_sample):
        X = data_sample["data_tensor"]
        metadata = data_sample["metainfo"]

        # NOTE: this function is called within inference_context manager
        #       see training/loops.py for details
        pred_hypercubes = self._predict(X, data_sample)

        buckets = self._build_pred_buckets(pred_hypercubes, metadata)
        keys_recv, coords_recv, pred_hypercubes_recv = self._alltoall(
            buckets, metadata, out_channels=self.num_output_channels
        )
        done_keys = self._apply_recv(keys_recv, coords_recv, pred_hypercubes_recv, data_type=self.main_output_name)

        if self.save_auxiliary_outputs:
            for aux_output, aux_metadata in self.auxiliary_outputs.items():
                aux_pred_hypercubes = data_sample[aux_output]
                # NOTE: might need to generalize _build_pred_buckets
                aux_buckets = self._build_pred_buckets(aux_pred_hypercubes, metadata)
                aux_keys_recv, aux_coords_recv, aux_pred_hypercubes_recv = self._alltoall(
                    aux_buckets, metadata, out_channels=aux_metadata["num_output_channels"]
                )
                self._apply_recv(aux_keys_recv, aux_coords_recv, aux_pred_hypercubes_recv, data_type=aux_output)

    def finalize(self):
        for key in list(self._tile_state.keys()):
            self._finish_if_done(key, force=True)
        barrier()
