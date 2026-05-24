from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time
from collections.abc import Mapping

from yolo_compare.logging_utils import clean_log_text


def run_logged(
    command: list[str],
    cwd: Path,
    log_path: Path,
    env: Mapping[str, str] | None = None,
    echo: bool = True,
) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process_env = os.environ.copy()
    if env:
        process_env.update(env)

    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()

        start = time.perf_counter()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            if echo:
                print(line, end="")
            log.write(clean_log_text(line))

        return_code = process.wait()
        elapsed_seconds = time.perf_counter() - start
        log.write(f"\nElapsed seconds: {elapsed_seconds:.3f}\n\n")
        if return_code != 0:
            raise SystemExit(f"Command failed with exit status {return_code}: {' '.join(command)}")
        return elapsed_seconds
