
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

SANDBOX_DIR = SCRIPT_DIR / "sandbox"
POSTGRES_SANDBOX_IMAGE = "docker://postgres:17"
MY_POSTGRES_CONF = SCRIPT_DIR / "my-postgres.conf"
SANDBOX_POSTGRES_CONF = SANDBOX_DIR / "etc" / "postgresql" / "postgresql.conf"


def load_repo_dotenv() -> None:
    _load_dotenv(REPO_ROOT / ".env")


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, _, rest = line.partition("=")
        key = key.strip()
        if not key:
            continue
        val = rest.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        os.environ[key] = val


def prepare_environment() -> None:
    _load_dotenv(REPO_ROOT / ".env")

    _KEEPALIVE_PARAMS = (
        "&keepalives=1&keepalives_idle=10"
        "&keepalives_interval=10&keepalives_count=60"
    )
    if not os.environ.get("PGCOPYDB_SOURCE_PGURI") and os.environ.get(
        "SUPABASE_PROD_ID"
    ) and os.environ.get("PGPASSWORD"):
        pid = os.environ["SUPABASE_PROD_ID"]
        os.environ["PGCOPYDB_SOURCE_PGURI"] = (
            f"postgresql://postgres@db.{pid}.supabase.co:5432/postgres"
            f"?sslmode=disable{_KEEPALIVE_PARAMS}"
        )

    os.environ.setdefault(
        "PGCOPYDB_TARGET_PGURI",
        "postgresql://postgres:postgres@127.0.0.1:5433/postgres"
        f"?{_KEEPALIVE_PARAMS.lstrip('&')}",
    )
    os.environ.setdefault(
        "PGCOPYDB_WORK_DIR", str(SCRIPT_DIR / "pgcopydb_work")
    )
    os.environ.setdefault("PGCOPYDB_SLOT_NAME", "cell_observatory_sandbox")
    os.environ.setdefault("PGCOPYDB_ORIGIN", "cell_observatory_sandbox")
    os.environ.setdefault(
        "PGCOPYDB_FILTER_FILE", str(SCRIPT_DIR / "filter.ini")
    )

    Path(os.environ["PGCOPYDB_WORK_DIR"]).mkdir(parents=True, exist_ok=True)


def pgcopydb_binary() -> str:
    exe = shutil.which("pgcopydb")
    if not exe:
        print(
            "pgcopydb not found in PATH. Install it or use the project Docker image "
            "(postgresql-client-17 + pgcopydb).",
            file=sys.stderr,
        )
        sys.exit(1)
    return exe


def require_source_uri() -> None:
    if not os.environ.get("PGCOPYDB_SOURCE_PGURI"):
        print(
            "Set PGCOPYDB_SOURCE_PGURI, or set SUPABASE_PROD_ID and PGPASSWORD in .env.",
            file=sys.stderr,
        )
        print(
            "Default URI uses the shared session pooler (Connect → Session mode); "
            "optional SUPABASE_POOLER_HOST / SUPABASE_DIRECT_DB_HOST — see scripts/db/sandbox.md.",
            file=sys.stderr,
        )
        sys.exit(1)


def run_pgcopydb(args: list[str]) -> int:
    cmd = [pgcopydb_binary(), *args]
    return subprocess.run(cmd, check=False).returncode


def _sync_user_stopped(rc: int) -> bool:
    """True if pgcopydb follow exited due to SIGINT/SIGTERM (common when stopping CDC)."""
    if rc < 0:
        return -rc in (2, 15)
    if rc >= 128:
        return (rc - 128) in (2, 15)
    return False


def _run_pgcopydb_copy_sequences(pgcopydb_extra: list[str]) -> int:
    args = [
        "copy",
        "sequences",
        "--source",
        os.environ["PGCOPYDB_SOURCE_PGURI"],
        "--target",
        os.environ["PGCOPYDB_TARGET_PGURI"],
        "--dir",
        os.environ["PGCOPYDB_WORK_DIR"],
        "--filters",
        os.environ["PGCOPYDB_FILTER_FILE"],
        "--not-consistent",
        *_pg_extra(pgcopydb_extra),
    ]
    return run_pgcopydb(args)


def _sql_quote_literal(val: str) -> str:
    return "'" + val.replace("'", "''") + "'"


def _terminate_slot_backend(source_uri: str, slot_name: str) -> int:
    """Best-effort: terminate backend currently using the given replication slot."""
    slot_lit = _sql_quote_literal(slot_name)
    sql = (
        "SELECT pg_terminate_backend(active_pid) "
        "FROM pg_replication_slots "
        f"WHERE slot_name = {slot_lit} AND active_pid IS NOT NULL;"
    )
    return subprocess.run(
        ["psql", source_uri, "-v", "ON_ERROR_STOP=1", "-c", sql],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode


def _clear_local_cdc_resume_state(work_dir: str) -> None:
    """Remove local pgcopydb CDC state so a new slot starts cleanly."""
    work_path = Path(work_dir)
    cdc_dir = work_path / "cdc"
    if cdc_dir.exists():
        shutil.rmtree(cdc_dir, ignore_errors=True)

    for name in ("snapshot", "pgcopydb.pid"):
        p = work_path / name
        if p.exists():
            p.unlink()


def _psql_scalar(uri: str, sql: str) -> str | None:
    proc = subprocess.run(
        ["psql", uri, "-tA", "-v", "ON_ERROR_STOP=1", "-c", sql],
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    if not out:
        return None
    return out.splitlines()[-1].strip()


def _slot_exists(source_uri: str, slot_name: str) -> bool:
    slot_lit = _sql_quote_literal(slot_name)
    sql = (
        "SELECT 1 "
        "FROM pg_replication_slots "
        f"WHERE slot_name = {slot_lit} "
        "LIMIT 1;"
    )
    return _psql_scalar(source_uri, sql) == "1"


def _drop_slot(source_uri: str, slot_name: str) -> int:
    slot_lit = _sql_quote_literal(slot_name)
    sql = f"SELECT pg_drop_replication_slot({slot_lit});"
    return subprocess.run(
        ["psql", source_uri, "-v", "ON_ERROR_STOP=1", "-c", sql],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode


def _has_local_cdc_state(work_dir: str) -> bool:
    work_path = Path(work_dir)
    return (
        (work_path / "cdc").exists()
        or (work_path / "snapshot").exists()
        or (work_path / "pgcopydb.pid").exists()
    )


def _origin_exists(target_uri: str, origin_name: str) -> bool:
    origin_lit = _sql_quote_literal(origin_name)
    sql = (
        "SELECT 1 "
        "FROM pg_replication_origin "
        f"WHERE roname = {origin_lit} "
        "LIMIT 1;"
    )
    return _psql_scalar(target_uri, sql) == "1"


def _drop_origin(target_uri: str, origin_name: str) -> int:
    origin_lit = _sql_quote_literal(origin_name)
    sql = f"SELECT pg_replication_origin_drop({origin_lit});"
    return subprocess.run(
        ["psql", target_uri, "-v", "ON_ERROR_STOP=1", "-c", sql],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode


def _slot_flush_reached(source_uri: str, slot_name: str, end_lsn: str) -> bool:
    slot_lit = _sql_quote_literal(slot_name)
    end_lit = _sql_quote_literal(end_lsn)
    sql = (
        "SELECT CASE "
        f"WHEN confirmed_flush_lsn >= {end_lit}::pg_lsn THEN 1 "
        "ELSE 0 END "
        "FROM pg_replication_slots "
        f"WHERE slot_name = {slot_lit} "
        "LIMIT 1;"
    )
    return _psql_scalar(source_uri, sql) == "1"


def _current_source_lsn(source_uri: str) -> str | None:
    return _psql_scalar(source_uri, "SELECT pg_current_wal_lsn();")


def _current_source_lsn_plus(source_uri: str, bytes_ahead: int) -> str | None:
    return _psql_scalar(
        source_uri, f"SELECT pg_current_wal_lsn() + {int(bytes_ahead)};"
    )


def _run_follow_one_shot_with_guard(
    args: list[str],
    *,
    source_uri: str,
    slot_name: str,
    end_lsn: str,
    timeout_seconds: int = 25,
) -> int:
    cmd = [pgcopydb_binary(), *args]
    proc = subprocess.Popen(cmd)
    try:
        return proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        if _slot_flush_reached(source_uri, slot_name, end_lsn):
            print(
                "sync: reached end LSN but pgcopydb follow did not exit; "
                "stopping follow process.",
                flush=True,
            )
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            return 0
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        return 124


def _require_apptainer() -> str:
    exe = shutil.which("apptainer")
    if not exe:
        print("apptainer not found in PATH.", file=sys.stderr)
        sys.exit(1)
    return exe


def _postgres_build_binds(cli_binds: list[str]) -> list[str]:
    load_repo_dotenv()
    raw = os.environ.get("SANDBOX_APPTAINER_BUILDBIND", "").strip()
    env_binds = shlex.split(raw) if raw else []
    return [*env_binds, *cli_binds]


def build(*, cli_binds: list[str]) -> int:
    _require_apptainer()
    if not MY_POSTGRES_CONF.is_file():
        print(f"build: missing {MY_POSTGRES_CONF}", file=sys.stderr)
        return 1
    binds = _postgres_build_binds(cli_binds)
    cmd: list[str] = ["apptainer", "build"]
    for spec in binds:
        cmd.extend(["--bind", spec])
    cmd.extend(
        ["-F", "--sandbox", str(SANDBOX_DIR.resolve()), POSTGRES_SANDBOX_IMAGE]
    )
    print(f"build: {shlex.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode
    if rc != 0:
        print(f"build: apptainer build failed (exit {rc})", file=sys.stderr)
        return rc
    if not SANDBOX_POSTGRES_CONF.parent.is_dir():
        print(
            f"build: expected {SANDBOX_POSTGRES_CONF.parent} after build",
            file=sys.stderr,
        )
        return 1
    shutil.copy2(MY_POSTGRES_CONF, SANDBOX_POSTGRES_CONF)
    print(
        f" build: installed {MY_POSTGRES_CONF.name} -> {SANDBOX_POSTGRES_CONF}",
        flush=True,
    )
    return 0


def run() -> int:
    _require_apptainer()
    if not SANDBOX_DIR.is_dir():
        print(
            f"Sandbox not found: {SANDBOX_DIR} (run build first)",
            file=sys.stderr,
        )
        return 1
    cmd = _apptainer_run_cmd()
    return subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode


def _apptainer_run_cmd() -> list[str]:
    return [
        "apptainer",
        "run",
        "--writable",
        "--pwd",
        "/var/lib/postgresql",
        "--env",
        "POSTGRES_PASSWORD=postgres",
        str(SANDBOX_DIR.resolve()),
        "-c",
        "port=5433",
        "-c",
        "config_file=/etc/postgresql/postgresql.conf",
    ]


def _target_postgres_ready(target_uri: str) -> bool:
    return _psql_scalar(target_uri, "SELECT 1;") == "1"


def _start_target_postgres_for_copy() -> subprocess.Popen[bytes] | None:
    target_uri = os.environ["PGCOPYDB_TARGET_PGURI"]
    if _target_postgres_ready(target_uri):
        print("Target Postgres already running on 5433; reusing it.", flush=True)
        return None

    _require_apptainer()
    if not SANDBOX_DIR.is_dir():
        raise RuntimeError(f"Sandbox not found: {SANDBOX_DIR} (run build first)")

    print("Starting sandbox Postgres via Apptainer for this command...", flush=True)
    proc = subprocess.Popen(
        _apptainer_run_cmd(),
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.time() + 45
    while time.time() < deadline:
        if _target_postgres_ready(target_uri):
            print("Sandbox Postgres is ready on 127.0.0.1:5433.", flush=True)
            return proc
        rc = proc.poll()
        if rc is not None:
            raise RuntimeError(
                f"Failed to start sandbox Postgres via Apptainer (exit {rc})."
            )
        time.sleep(0.5)

    proc.terminate()
    raise RuntimeError("Timed out waiting for sandbox Postgres on 127.0.0.1:5433.")


def _stop_started_target_postgres(proc: subprocess.Popen[bytes] | None) -> None:
    if not proc:
        return
    if proc.poll() is not None:
        return

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    print("Stopped auto-started sandbox Postgres.", flush=True)


def _archive_date_str() -> str:
    return os.environ.get("SANDBOX_ARCHIVE_DATE", date.today().strftime("%Y_%m_%d"))


def run_archive() -> int:
    load_repo_dotenv()
    if not SANDBOX_DIR.is_dir():
        print(f"Sandbox not found: {SANDBOX_DIR}", file=sys.stderr)
        return 1

    db_dir = os.environ.get("DATABASE_DIR", "").strip()
    if not db_dir:
        print(
            "archive: set DATABASE_DIR (repo .env or environment) "
            "to copy the tarball under DATABASE_DIR/YYYY_MM_DD/sandbox.tar.zst.",
            file=sys.stderr,
        )
        return 1

    d = _archive_date_str()
    out = SCRIPT_DIR / f"{d}_sandbox.tar.zst"
    print(f"archive: writing {out}", flush=True)

    shell_cmd = (
        f"tar --warning=no-file-changed "
        f"-I 'zstd -3 -T0' -cvf {shlex.quote(str(out))} "
        f"-C {shlex.quote(str(SCRIPT_DIR))} sandbox"
    )
    rc = subprocess.run(shell_cmd, shell=True, check=False).returncode
    if rc > 1:
        print(f"archive: tar failed (exit {rc})", file=sys.stderr)
        return rc

    dest_parent = Path(db_dir).expanduser() / d
    dest_parent.mkdir(parents=True, exist_ok=True)
    dest = dest_parent / "sandbox.tar.zst"
    print(f"archive: copying to {dest}", flush=True)
    shutil.copy2(out, dest)
    return 0


def _pg_extra(unknown: list[str]) -> list[str]:
    if unknown and unknown[0] == "--":
        return list(unknown[1:])
    return list(unknown)


def snapshot(pgcopydb_extra: list[str]) -> int:
    prepare_environment()
    pgcopydb_binary()
    require_source_uri()
    started_target: subprocess.Popen[bytes] | None = None

    try:
        started_target = _start_target_postgres_for_copy()
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    try:
        work = os.environ["PGCOPYDB_WORK_DIR"]
        filt = os.environ["PGCOPYDB_FILTER_FILE"]
        print(f"Work dir: {work}")
        print(f"Filter:   {filt}")

        args = [
            "clone",
            "--source",
            os.environ["PGCOPYDB_SOURCE_PGURI"],
            "--target",
            os.environ["PGCOPYDB_TARGET_PGURI"],
            "--dir",
            work,
            "--filters",
            filt,
            "--no-owner",
            "--no-acl",
            "--drop-if-exists",
            "--table-jobs",
            "8",
            *_pg_extra(pgcopydb_extra),
        ]
        return run_pgcopydb(args)
    finally:
        _stop_started_target_postgres(started_target)


def _check_direct_connection(label: str) -> None:
    uri = os.environ.get("PGCOPYDB_SOURCE_PGURI", "")
    if "pooler.supabase.com" in uri:
        print(
            f"ERROR: {label} requires a *direct* connection "
            f"(db.<ref>.supabase.co), but PGCOPYDB_SOURCE_PGURI points at "
            f"the session-mode pooler.  The pooler does not forward the "
            f"PostgreSQL replication protocol.\n"
            f"Set PGCOPYDB_SOURCE_PGURI to a direct URI, or use `snapshot` "
            f"instead (works through the pooler).",
            file=sys.stderr,
        )
        sys.exit(1)


_TARGET_EXTENSIONS = ["intarray"]
_PREPARE_ROLES_SQL = SCRIPT_DIR / "prepare_roles.sql"


def _prepare_target() -> None:
    target = os.environ["PGCOPYDB_TARGET_PGURI"]

    if _PREPARE_ROLES_SQL.is_file():
        rc = subprocess.run(
            ["psql", target, "-f", str(_PREPARE_ROLES_SQL)],
            check=False,
        ).returncode
        if rc != 0:
            print(
                f"WARNING: prepare_roles.sql failed (exit {rc}); "
                "RLS policy restore may have errors.",
                file=sys.stderr,
            )

    for ext in _TARGET_EXTENSIONS:
        cmd = [
            "psql", target, "-c",
            f"CREATE EXTENSION IF NOT EXISTS {ext} SCHEMA public;",
        ]
        rc = subprocess.run(cmd, check=False).returncode
        if rc != 0:
            print(
                f"WARNING: could not create extension {ext} on target "
                f"(exit {rc}); schema restore may have errors.",
                file=sys.stderr,
            )


def clone(pgcopydb_extra: list[str]) -> int:
    prepare_environment()
    pgcopydb_binary()
    require_source_uri()
    _check_direct_connection("clone")
    started_target: subprocess.Popen[bytes] | None = None

    try:
        started_target = _start_target_postgres_for_copy()
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    try:
        work_path = Path(os.environ["PGCOPYDB_WORK_DIR"])
        if work_path.exists():
            print(f"Removing stale work dir: {work_path}")
            shutil.rmtree(work_path)
        work_path.mkdir(parents=True, exist_ok=True)

        _prepare_target()

        work = str(work_path)
        filt = os.environ["PGCOPYDB_FILTER_FILE"]
        print(f"Work dir: {work}")
        print(f"Filter:   {filt}")

        args = [
            "clone",
            "--source",
            os.environ["PGCOPYDB_SOURCE_PGURI"],
            "--target",
            os.environ["PGCOPYDB_TARGET_PGURI"],
            "--dir",
            work,
            "--filters",
            filt,
            "--no-owner",
            "--no-acl",
            "--skip-extensions",
            "--drop-if-exists",
            "--table-jobs",
            "8",
            *_pg_extra(pgcopydb_extra),
        ]
        return run_pgcopydb(args)
    finally:
        _stop_started_target_postgres(started_target)


def sync(
    pgcopydb_extra: list[str],
    *,
    copy_sequences_after: bool = True,
    continuous: bool = False,
) -> int:
    prepare_environment()
    pgcopydb_binary()
    require_source_uri()
    _check_direct_connection("sync")
    started_target: subprocess.Popen[bytes] | None = None

    try:
        started_target = _start_target_postgres_for_copy()
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    try:
        source_uri = os.environ["PGCOPYDB_SOURCE_PGURI"]
        target_uri = os.environ["PGCOPYDB_TARGET_PGURI"]
        work = os.environ["PGCOPYDB_WORK_DIR"]
        slot = os.environ["PGCOPYDB_SLOT_NAME"]
        origin = os.environ["PGCOPYDB_ORIGIN"]
        filters = os.environ["PGCOPYDB_FILTER_FILE"]
        extra = _pg_extra(pgcopydb_extra)
        print(f"Work dir: {work}")
        print(f"Slot:     {slot}")

        snap_path = Path(work) / "snapshot"
        if snap_path.is_file():
            print(f"Removing clone snapshot marker for follow: {snap_path}")
            snap_path.unlink()

        args = [
            "follow",
            "--source",
            source_uri,
            "--target",
            target_uri,
            "--dir",
            work,
            "--filters",
            filters,
            "--slot-name",
            slot,
            "--origin",
            origin,
            "--not-consistent",
        ]
        if _slot_exists(source_uri, slot):
            print(
                f"Resetting existing replication slot before sync: {slot}",
                flush=True,
            )
            _terminate_slot_backend(source_uri, slot)
            if _drop_slot(source_uri, slot) != 0:
                print(
                    f"sync: failed to reset replication slot {slot}; "
                    "run ./sandbox_cli cleanup and retry.",
                    file=sys.stderr,
                )
                return 1
            _clear_local_cdc_resume_state(work)
        elif _has_local_cdc_state(work):
            print(
                "Clearing stale local CDC resume state (slot not present on source).",
                flush=True,
            )
            _clear_local_cdc_resume_state(work)

        if _origin_exists(target_uri, origin):
            print(
                f"Resetting existing replication origin before sync: {origin}",
                flush=True,
            )
            if _drop_origin(target_uri, origin) != 0:
                print(
                    f"sync: failed to reset replication origin {origin} on target.",
                    file=sys.stderr,
                )
                return 1
        args.append("--create-slot")

        end_lsn: str | None = None
        if continuous:
            print("Mode:     continuous follow")
        elif "--endpos" not in extra:
            end_lsn = _current_source_lsn_plus(source_uri, 56)
            if not end_lsn:
                end_lsn = _current_source_lsn(source_uri)
            if not end_lsn:
                print(
                    "sync: failed to fetch source pg_current_wal_lsn(); "
                    "cannot compute one-shot catch-up end position.",
                    file=sys.stderr,
                )
                return 1
            print(f"End LSN:  {end_lsn} (one-shot catch-up)")
            args.extend(["--endpos", end_lsn])

        args.extend(extra)
        if continuous:
            rc_follow = run_pgcopydb(args)
        elif end_lsn:
            rc_follow = _run_follow_one_shot_with_guard(
                args,
                source_uri=source_uri,
                slot_name=slot,
                end_lsn=end_lsn,
            )
        else:
            rc_follow = run_pgcopydb(args)
        if not copy_sequences_after:
            return rc_follow
        if rc_follow != 0 and not _sync_user_stopped(rc_follow):
            return rc_follow

        rc_seq = _run_pgcopydb_copy_sequences([])
        if rc_seq != 0:
            return rc_seq
        return 0
    finally:
        _stop_started_target_postgres(started_target)


def cleanup(pgcopydb_extra: list[str]) -> int:
    prepare_environment()
    pgcopydb_binary()
    require_source_uri()

    work = os.environ["PGCOPYDB_WORK_DIR"]
    Path(work).mkdir(parents=True, exist_ok=True)

    source_uri = os.environ["PGCOPYDB_SOURCE_PGURI"]
    slot_name = os.environ["PGCOPYDB_SLOT_NAME"]

    args = [
        "stream",
        "cleanup",
        "--source",
        source_uri,
        "--target",
        os.environ["PGCOPYDB_TARGET_PGURI"],
        "--dir",
        work,
        "--slot-name",
        slot_name,
        "--origin",
        os.environ["PGCOPYDB_ORIGIN"],
        *_pg_extra(pgcopydb_extra),
    ]

    _terminate_slot_backend(source_uri, slot_name)
    rc = run_pgcopydb(args)
    if rc == 0:
        _clear_local_cdc_resume_state(work)
        return 0

    _terminate_slot_backend(source_uri, slot_name)
    time.sleep(0.25)
    rc = run_pgcopydb(args)
    if rc == 0:
        _clear_local_cdc_resume_state(work)
    return rc


def archive() -> int:
    return run_archive()
