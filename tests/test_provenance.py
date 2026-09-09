"""Run provenance: what produced a run, without calling git."""

import json
import subprocess
from pathlib import Path

from lettuce_greenhouse_gym.provenance import git_commit, run_metadata, write_run_metadata

REPO = Path(__file__).parents[1]


def test_git_commit_matches_git_when_available():
    ours = git_commit(REPO)
    try:
        theirs = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return  # no git here; the reading side is covered by the packed-refs test below
    assert ours == theirs


def test_git_commit_reads_loose_and_packed_refs_and_detached_heads(tmp_path):
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "refs" / "heads" / "main").write_text("a" * 40 + "\n")
    (tmp_path / "some" / "nested").mkdir(parents=True)
    assert git_commit(tmp_path / "some" / "nested") == "a" * 40
    (git / "refs" / "heads" / "main").unlink()
    (git / "packed-refs").write_text("# pack-refs\n" + "b" * 40 + " refs/heads/main\n")
    assert git_commit(tmp_path) == "b" * 40
    (git / "HEAD").write_text("c" * 40 + "\n")
    assert git_commit(tmp_path) == "c" * 40
    assert git_commit(Path("/")) is None


def test_run_metadata_and_file_round_trip(tmp_path):
    meta = run_metadata(seed=3)
    assert meta["package"] == "lettuce-greenhouse-gym"
    assert meta["dependencies"]["numpy"]
    assert meta["seed"] == 3
    path = write_run_metadata(tmp_path)
    write_run_metadata(tmp_path, finished="later")
    data = json.loads(path.read_text())
    assert data["started"]
    assert data["finished"] == "later"
    assert data["argv"] == meta["argv"]
