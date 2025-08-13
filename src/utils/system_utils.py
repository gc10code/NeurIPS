import random
import numpy as np
import torch
from packaging import version
import psutil
import shutil
from typing import Dict
from pathlib import Path
import logging
from typing import Optional
from src.utils.logging import lprint, LoggingLevels as ll

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_pytorch_version(min_version: str = "1.9.0") -> None:
    if version.parse(torch.__version__) < version.parse(min_version):
        raise RuntimeError(f"PyTorch version {torch.__version__} is too old. Required: >= {min_version}")

def monitor_memory() -> Dict[str, float]:
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss_gb': memory_info.rss / (1024**3),
            'vms_gb': memory_info.vms / (1024**3),
            'percent': process.memory_percent()
        }
    except psutil.Error as e:
        logger = logging.getLogger('RPropMLP')
        lprint(ll.WARN,  f"Memory monitoring failed: {str(e)}")
        return {'rss_gb': -1, 'vms_gb': -1, 'percent': -1}
    

def setup_device(device: Optional[torch.device] = torch.device('cpu')) -> torch.device:
    """Set up the computation device (CPU or GPU)."""
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            lprint(ll.INFO,  f"Using GPU: {torch.cuda.get_device_name()}, Memory: {total_memory:.1f} GB")
        except RuntimeError as e:
            lprint(ll.WARN,  f"GPU detection failed: {str(e)}. Falling back to CPU.")
            device = torch.device('cpu')
    lprint(ll.INFO,  f"Using device: {device}")
    return device

def remove_dir(path:Path):
    if path.exists() and path.is_dir():
        shutil.rmtree(path)

def signal_handler(sig, frame):
    lprint(ll.EXIT, "Received interrupt signal, shutting down")
    raise SystemExit()