"""What produced a result: the source, the solver build, and the environment."""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


class RunProvenance:
    """Records what a result was produced by, so a reader can check it.

    A commit hash only identifies a run inside the history it was made in, and
    results are often republished into a fresh one. The fingerprint covers the
    source files themselves instead, so it can be recomputed from any
    checkout, with or without git.
    """

    def __init__(self, repo_root: Path | None = None):
        # This module lives in src/, so the repository root is one level up.
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[1])

    def package_version(self, name: str) -> str:
        """Return an installed package's version, or 'unknown'."""
        try:
            from importlib.metadata import PackageNotFoundError, version

            return version(name)
        except (PackageNotFoundError, ImportError, ValueError):
            return "unknown"

    def ipopt_version(self) -> str:
        """Return the Ipopt build string, which differs between installations."""
        executable = shutil.which("ipopt")
        if not executable:
            return "not found on PATH"
        try:
            out = subprocess.run(  # noqa: S603  (fixed argv, resolved executable)
                [executable, "--version"], capture_output=True, text=True, timeout=30
            )
            return (out.stdout or out.stderr).strip().splitlines()[0]
        except (OSError, subprocess.SubprocessError, IndexError):
            return "unknown"

    def code_fingerprint(self) -> str:
        """Return a hash of the source that produced a result.

        A git commit only identifies a run inside the repository it was made in.
        Results are often republished into a fresh history, where that commit does
        not exist, so the hash covers the source files themselves: anyone holding
        the code and the results can recompute it and see whether they match,
        with or without git.
        """
        digest = hashlib.sha256()
        roots = [self.repo_root / "src", self.repo_root / "converters"]
        files = [self.repo_root / "run_curtailment_study.py"]
        for root in roots:
            files.extend(sorted(root.rglob("*.py")))
        for path in sorted(
            files, key=lambda item: str(item.relative_to(self.repo_root))
        ):
            if not path.is_file():
                continue
            digest.update(str(path.relative_to(self.repo_root)).encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def describe(self) -> dict:
        """Describe what produced a result, so a run can be reproduced or explained.

        A curtailment number is only meaningful alongside the solver that produced
        it: this study has already seen two Ipopt builds disagree, and a converter
        version silently change a network. Recording the versions, the commit and
        the hardware makes a rerun comparable rather than merely similar.
        """
        return {
            "code_sha256": self.code_fingerprint(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
            "cpu_count": os.cpu_count(),
            "ipopt": self.ipopt_version(),
            "packages": {
                name: self.package_version(name)
                for name in ("pyomo", "numpy", "matplotlib", "powerio")
            },
            "thread_limits": {
                var: os.environ.get(var)
                for var in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        }
