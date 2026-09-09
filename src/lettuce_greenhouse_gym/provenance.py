"""What produced a run: versions, platform, invocation, and the source commit if there is one.

Written by :func:`lettuce_greenhouse_gym.train.sb3.train` as ``run.json`` next to ``spec.yaml``. The
spec says *what* was run; this says *with which build*, so a result can be traced to code.
"""

import json
import platform
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

DEPENDENCIES = ("casadi", "gymnasium", "numpy", "stable-baselines3", "torch", "wandb")


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def git_commit(path: str | Path = ".") -> str | None:
    """The commit ``HEAD`` points to in the repository containing ``path``, read from ``.git`` itself.

    Returns ``None`` when ``path`` is not inside a git checkout. Reads the files git writes rather
    than calling git, so it needs no executable and cannot run anything; it does not report whether
    the tree had uncommitted changes.
    """
    here = Path(path).resolve()
    for directory in (here, *here.parents):
        git_dir = directory / ".git"
        if git_dir.is_file():  # a worktree: the file names the real git directory
            pointer = git_dir.read_text().strip()
            git_dir = Path(pointer.removeprefix("gitdir:").strip())
            if not git_dir.is_absolute():
                git_dir = directory / git_dir
        if not git_dir.is_dir():
            continue
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head  # detached HEAD: the hash itself
        ref = head.removeprefix("ref:").strip()
        loose = git_dir / ref
        if loose.exists():
            return loose.read_text().strip()
        # the common git dir holds packed refs (and, for a worktree, the branch refs)
        common = git_dir / "commondir"
        base = (git_dir / common.read_text().strip()).resolve() if common.exists() else git_dir
        for candidate in (base / ref, base / "packed-refs"):
            if not candidate.exists():
                continue
            if candidate.name != "packed-refs":
                return candidate.read_text().strip()
            for line in candidate.read_text().splitlines():
                if line.endswith(" " + ref):
                    return line.split(" ", 1)[0]
        return None
    return None


def run_metadata(**extra: Any) -> dict[str, Any]:
    """Package and dependency versions, interpreter, platform, argv, working directory, commit."""
    return {
        "package": "lettuce-greenhouse-gym",
        "version": _version("lettuce-greenhouse-gym"),
        "dependencies": {d: _version(d) for d in DEPENDENCIES},
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "git_commit": git_commit(),
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        **extra,
    }


def write_run_metadata(run_dir: str | Path, **extra: Any) -> Path:
    """Write ``run.json`` into ``run_dir``; call again with more fields to extend it."""
    path = Path(run_dir) / "run.json"
    data = json.loads(path.read_text()) if path.exists() else run_metadata()
    data.update(extra)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")
    return path
