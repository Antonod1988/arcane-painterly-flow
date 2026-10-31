"""Run a Blender script headless while holding the shared GPU lock, so only one heavy render runs at a time.

    python src/gpu_run.py <script.py> [script args...]

The lock is an OS byte-range lock on work/gpu.lock: it is released automatically if this process dies.
Set BLENDER to the Blender executable if it is not on PATH.
"""
import os
import subprocess
import sys
import time

try:
    import msvcrt

    def _try_lock(f):
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(f):
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
except ImportError:                                   # Linux / macOS
    import fcntl

    def _try_lock(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")   # repository root

BLENDER = os.environ.get("BLENDER", "blender")
LOCK = ROOT + "/work/gpu.lock"


def main():
    script, args = sys.argv[1], sys.argv[2:]
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    f = open(LOCK, "a+")
    waited = 0.0
    while True:
        try:
            _try_lock(f)
            break
        except OSError:
            if waited == 0:
                print("waiting for the GPU lock ...", flush=True)
            time.sleep(3)
            waited += 3
    t0 = time.time()
    print(f"GPU lock taken after {waited:.0f}s: {os.path.basename(script)} {' '.join(args)}", flush=True)
    try:
        r = subprocess.run([BLENDER, "-b", "--factory-startup", "--python", script, "--", *args])
    finally:
        _unlock(f)
        f.close()
    print(f"blender exit {r.returncode} after {time.time() - t0:.0f}s", flush=True)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
