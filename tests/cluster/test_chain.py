"""LSF job chaining: TRAINING_DONE marker, follower script, and the bash gate
(cluster/chain_lib.sh) exercised with a fake bsub."""
import os
import subprocess
from pathlib import Path

import pytest
import ujson
from omegaconf import OmegaConf

from cell_observatory_platform.manager import write_chain_resubmit_script
from cell_observatory_platform.training.helpers import write_training_done

REPO = Path(__file__).resolve().parents[2]
CHAIN_LIB = REPO / "cluster" / "chain_lib.sh"


def test_write_training_done_writes_json_marker(tmp_path):
    cfg = OmegaConf.create({"paths": {"outdir": str(tmp_path)}})
    marker = write_training_done(cfg, iter=1234, epoch=10)
    assert marker == tmp_path / "TRAINING_DONE"
    payload = ujson.loads(marker.read_text())
    assert payload["iter"] == 1234 and payload["epoch"] == 10 and "time" in payload
    assert not (tmp_path / "TRAINING_DONE.tmp").exists()


def test_write_training_done_without_outdir_is_noop():
    assert write_training_done(OmegaConf.create({"paths": {"outdir": None}}), 1, 1) is None
    assert write_training_done(OmegaConf.create({}), 1, 1) is None


def test_chain_resubmit_script_holds_on_parent_and_decrements(tmp_path):
    script = write_chain_resubmit_script(
        str(tmp_path), ["bsub", "-q", "b300", "-n", "96", "-J", "run1"], "bash ray_lsf_cluster.sh -o x", 1200
    )
    text = script.read_text()
    assert text.startswith("#!/usr/bin/env bash")
    assert os.access(script, os.X_OK)
    assert '-w "ended(${PARENT_JOBID})"' in text
    assert 'CHAIN_REMAINING=${CHAIN_REMAINING}' in text
    assert "CHAIN_MIN_RUNTIME=${CHAIN_MIN_RUNTIME:-1200}" in text
    assert "bsub -q b300 -n 96 -J run1" in text


# --------------------------------------------------------------- bash gate --
def _fake_bsub(tmp_path: Path) -> Path:
    """A bsub on PATH that records its argv + chain env and exits 0."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    fake = bindir / "bsub"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$@" >> "{tmp_path}/bsub_calls"\n'
        f'echo "CHAIN_REMAINING=$CHAIN_REMAINING" >> "{tmp_path}/bsub_calls"\n'
        'echo "Job <999> is submitted"\n'
    )
    fake.chmod(0o755)
    return bindir


def _run_gate(tmp_path: Path, outdir: Path, env: dict, body: str = "echo body-ran") -> subprocess.CompletedProcess:
    """Source chain_lib.sh, run chain_job_start on outdir, then `body`, like the wrapper does."""
    script = (
        f'source "{CHAIN_LIB}"\n'
        f'chain_job_start "{outdir}"\n'
        f"{body}\n"
    )
    full_env = {"PATH": f"{_fake_bsub(tmp_path)}:{os.environ['PATH']}", "LSB_JOBID": "123", **env}
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=full_env)


def _write_follower(outdir: Path) -> None:
    write_chain_resubmit_script(str(outdir), ["bsub", "-q", "b300"], "bash wrapper.sh", 1200)


def test_gate_exits_without_running_when_marker_exists(tmp_path):
    outdir = tmp_path / "run"
    outdir.mkdir()
    _write_follower(outdir)
    (outdir / "TRAINING_DONE").write_text("{}")
    r = _run_gate(tmp_path, outdir, {"CHAIN_REMAINING": "3"})
    assert r.returncode == 0, r.stderr
    assert "body-ran" not in r.stdout and "run complete" in r.stdout
    assert not (tmp_path / "bsub_calls").exists()
    assert not (outdir / "chain_last_exit").exists()


def test_gate_honours_chain_stop(tmp_path):
    outdir = tmp_path / "run"
    outdir.mkdir()
    _write_follower(outdir)
    (outdir / "CHAIN_STOP").touch()
    r = _run_gate(tmp_path, outdir, {"CHAIN_REMAINING": "3"})
    assert r.returncode == 0 and "body-ran" not in r.stdout
    assert not (tmp_path / "bsub_calls").exists()


def test_fresh_link_submits_one_follower_and_records_exit(tmp_path):
    outdir = tmp_path / "run"
    outdir.mkdir()
    _write_follower(outdir)
    r = _run_gate(tmp_path, outdir, {"CHAIN_REMAINING": "2"}, body="echo body-ran; exit 7")
    assert r.returncode == 7, r.stderr
    assert "body-ran" in r.stdout
    calls = (tmp_path / "bsub_calls").read_text()
    assert calls.count("ended(123)") == 1, calls
    assert "-q b300" in calls and "CHAIN_REMAINING=1" in calls
    rc, runtime, jobid = (outdir / "chain_last_exit").read_text().split()
    assert (rc, jobid) == ("7", "123") and int(runtime) >= 0


def test_last_link_submits_nothing(tmp_path):
    outdir = tmp_path / "run"
    outdir.mkdir()
    _write_follower(outdir)
    r = _run_gate(tmp_path, outdir, {"CHAIN_REMAINING": "0"})
    assert r.returncode == 0 and "body-ran" in r.stdout and "last link" in r.stdout
    assert not (tmp_path / "bsub_calls").exists()
    assert (outdir / "chain_last_exit").exists()


def test_follower_stops_after_an_early_crash(tmp_path):
    outdir = tmp_path / "run"
    outdir.mkdir()
    _write_follower(outdir)
    (outdir / "chain_last_exit").write_text("1 45 122\n")  # crashed after 45 s
    r = _run_gate(tmp_path, outdir, {"CHAIN_REMAINING": "2"})
    assert r.returncode == 1 and "body-ran" not in r.stdout
    assert "stopping the chain" in r.stderr
    assert not (tmp_path / "bsub_calls").exists()


def test_follower_resumes_after_a_runlimit_kill(tmp_path):
    outdir = tmp_path / "run"
    outdir.mkdir()
    _write_follower(outdir)
    (outdir / "chain_last_exit").write_text("143 14300 122\n")  # killed by -W after ~4 h
    r = _run_gate(tmp_path, outdir, {"CHAIN_REMAINING": "1"})
    assert r.returncode == 0 and "body-ran" in r.stdout
    assert "resuming after job 122" in r.stdout
    assert "ended(123)" in (tmp_path / "bsub_calls").read_text()


def test_min_runtime_is_configurable(tmp_path):
    outdir = tmp_path / "run"
    outdir.mkdir()
    _write_follower(outdir)
    (outdir / "chain_last_exit").write_text("1 45 122\n")
    r = _run_gate(tmp_path, outdir, {"CHAIN_REMAINING": "1", "CHAIN_MIN_RUNTIME": "10"})
    assert r.returncode == 0 and "body-ran" in r.stdout
