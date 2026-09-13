# adds src to the path so the tests find the package without installing it

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
