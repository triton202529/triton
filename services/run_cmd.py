from __future__ import annotations

import subprocess
from typing import List, Tuple


def run_cmd(cmd: List[str], timeout: int = 300) -> Tuple[int, str]:
    """
    Run a command and capture stdout+stderr.
    Returns (returncode, combined_output).
    """
    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=False,
    )
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    return p.returncode, out.strip()
