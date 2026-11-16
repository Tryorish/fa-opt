"""Run profile_workload in this process, preserving output even if hipprof hides it."""

import faulthandler
import json
import os
from pathlib import Path
import runpy
import sys
import traceback


def main():
    log = Path(sys.argv[1])
    workload = Path(__file__).with_name("profile_workload.py").resolve()
    sys.argv = [str(workload), *sys.argv[2:]]
    # Redirect native stderr as well as Python output. Keep the same process
    # for HIP interception; no child process or extra GPU work is introduced.
    with log.open("w", buffering=1) as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        faulthandler.enable()
        print(json.dumps({"capture": "started", "python": sys.executable,
                          "pid": os.getpid(), "cwd": os.getcwd(),
                          "argv": sys.argv}), flush=True)
        status = 1
        try:
            runpy.run_path(str(workload), run_name="__main__")
            status = 0
        except SystemExit as error:
            status = error.code if isinstance(error.code, int) else (0 if error.code is None else 1)
            if status:
                traceback.print_exc()
        except BaseException:
            traceback.print_exc()
        finally:
            print(f"Capture exit status: {status}", flush=True)
            log.with_suffix(".status").write_text(f"{status}\n")
    return status


if __name__ == "__main__":
    sys.exit(main())
