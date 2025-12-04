import os
import sys
from pathlib import Path
from multiprocessing import shared_memory

SHM_DIR = Path("/dev/shm")
PREFIXES = ("psm_",)

def unlink_shared_memory() -> int:
    euid = os.geteuid()
    removed = 0

    if not SHM_DIR.is_dir():
        return 0

    for p in SHM_DIR.iterdir():
        name = p.name
        if not any(name.startswith(pref) for pref in PREFIXES):
            continue

        # try:
        #     st = p.stat()
        # except FileNotFoundError:
        #     continue

        # if st.st_uid != euid:
        #     continue

        try:
            shm = shared_memory.SharedMemory(name=name)
        except FileNotFoundError:
            print(f"Shared memory {name} not found when attempting to unlink.")
            continue
        except PermissionError:
            print(f"Permission denied when attempting to access shared memory {name}.")
            continue

        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
            removed += 1
        except FileNotFoundError:
            print(f"Shared memory {name} not found when attempting to unlink.")
            pass
        except PermissionError:
            print(f"Permission denied when attempting to access shared memory {name}.")
            pass

    return removed

if __name__ == "__main__":
    n = unlink_shared_memory()
    sys.exit(0)