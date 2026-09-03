from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Optional

import connectorx as cx
import pyarrow as pa
from dotenv import load_dotenv

from cell_observatory_platform.utils.context import node_ip

logger = logging.getLogger(__name__)


class LocalArrowDatabase:
    def __init__(
        self,
        dbname: str = "local",
        dotenv_path: Optional[str] = None,
        database_url: Optional[str] = None,
        protocol: str = "cursor",
        statement_timeout_ms: int = 600_000,
        verbose: bool = True,
    ) -> None:
        t0 = time.perf_counter()
        self.dbname = str(dbname)
        self.dotenv_path = str(dotenv_path) if dotenv_path is not None else None
        self.protocol = str(protocol)
        self.statement_timeout_ms = int(statement_timeout_ms)
        self.verbose = bool(verbose)
        if database_url:
            self.database_url = database_url
        else:
            if self.dotenv_path is not None:
                dotenv_file = Path(self.dotenv_path)
                if not dotenv_file.exists():
                    raise FileNotFoundError(f"dotenv_path={self.dotenv_path!r} does not exist")
                load_dotenv(dotenv_file, verbose=self.verbose, override=False)
            port = os.environ.get("SUPABASE_LOCAL_PORT")
            if not port:
                raise ValueError(
                    "SUPABASE_LOCAL_PORT must be set for dbname='local'. "
                    "The local database host is resolved from the current node at runtime."
                )
            host = node_ip()
            self.database_url = f"postgresql://postgres:postgres@{host}:{int(port)}/postgres"
            logger.info("[LocalArrowDatabase] resolved local URI: host=%s port=%s", host, port)

        if self.verbose:
            logger.info(
                "[LocalArrowDatabase] initialized dbname=%s protocol=%s timeout_ms=%s in %.2fs",
                self.dbname,
                self.protocol,
                self.statement_timeout_ms,
                time.perf_counter() - t0,
            )

    @staticmethod
    def _query_preview(sql: str, *, max_chars: int = 160) -> str:
        compact = " ".join(str(sql).split())
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 3] + "..."

    def execute_arrow(self, sql: str) -> pa.Table:
        t0 = time.perf_counter()
        preview = self._query_preview(sql)
        if self.verbose:
            logger.info("[LocalArrowDatabase] executing query: %s", preview)
        table = cx.read_sql(
            conn=self.database_url,
            query=sql,
            protocol=self.protocol,
            return_type="arrow",
            pre_execution_query=[f"SET statement_timeout = '{self.statement_timeout_ms}';"],
        )
        t1 = time.perf_counter()
        if self.verbose:
            logger.info(
                "[LocalArrowDatabase] query finished rows=%s elapsed=%.2fs sql=%s",
                table.num_rows,
                t1 - t0,
                preview,
            )
        return table

    def relation_exists(self, relation_name: str, *, schema: str = "public") -> bool:
        qualified_name = f"{schema}.{relation_name}"
        escaped_name = qualified_name.replace("'", "''")
        table = self.execute_arrow(
            f"SELECT to_regclass('{escaped_name}') IS NOT NULL AS relation_exists"
        )
        if table.num_rows != 1:
            raise ValueError(f"Expected 1 row when checking relation existence for {qualified_name!r}")
        return bool(table["relation_exists"][0].as_py())

    def assert_relation_exists(self, relation_name: str, *, schema: str = "public") -> None:
        if not self.relation_exists(relation_name, schema=schema):
            raise ValueError(
                f"Required relation {schema}.{relation_name} does not exist in dbname={self.dbname!r}"
            )
