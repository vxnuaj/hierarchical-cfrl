"""Put the vendored ELF source root on sys.path so its absolute imports
(`from utils...`, `from modules...`, `from configs...`) resolve."""
import sys
from pathlib import Path

ELF_ROOT = Path(__file__).resolve().parent / "elf"
if str(ELF_ROOT) not in sys.path:
    sys.path.insert(0, str(ELF_ROOT))
