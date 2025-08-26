from pathlib import Path
import pickle
import sys, os

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import pandas as pd
import numpy as np
from typing import List, Dict
from src.models.fusion import main_test_random

main_test_random()

