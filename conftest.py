# Make the src-layout packages importable in tests (the controller's hermetic verification
# run puts src/ on sys.path itself).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
