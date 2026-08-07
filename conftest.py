"""Put src/ on sys.path so tests run before `pip install -e .` is done.
Once you install the package editable, this is harmless/redundant.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
