"""Start a command in its own session, logging to a file, under `caffeinate -i`.

    python scripts/detach.py logs/<name>.log -- <command ...>

Runs started from Claude's task shell die with it (Oct 27: a training run did). This
detaches them (new session, parent launchd), so only `make pause-training` / a kill stops
them. Prints the PID; the process group id is the same.
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        sys.exit(__doc__)
    log = Path(sys.argv[1])
    log.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(["caffeinate", "-i", *sys.argv[3:]], stdout=open(log, "w"),
                               stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"started pid {process.pid}, log {log}")
