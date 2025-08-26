from pathlib import Path
from dataclasses import dataclass
from typing import List

@dataclass
class NeurIPSFiles:
    train_supplements: List[Path] 
    train: Path
    test: Path

