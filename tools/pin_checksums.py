# prints the sha256 of each data file so it can be pasted into config.py

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from charm_acp.config import PRODUCTION, file_digest, production_spec

names = sys.argv[1:] or [PRODUCTION]

for name in names:
    name, spec = production_spec(name)
    print(f'    "{name}": {{')
    print('        "sha256": {')
    for fname in spec["files"]:
        path = spec["dir"] / fname
        if not path.exists():
            print(f'            # MISSING ON DISK: {fname}')
            continue
        gb = path.stat().st_size / 1e9
        print(f'            "{fname}":', flush=True, end=" ")
        print(f'"{file_digest(path)}",   # {gb:.2f} GB', flush=True)
    print("        },")
    print("    },")
