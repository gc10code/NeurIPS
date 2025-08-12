import logging
import logging.handlers
from multiprocessing import Queue
from colorama import init, Fore, Style
from typing import Optional
from pathlib import Path

init(autoreset=True)

log_queue = Queue()

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.WHITE,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }

    def format(self, record):
        fold = getattr(record, 'fold', '')
        param_idx = getattr(record, 'param_idx', '')
        prefix = f"{fold}{': ' if fold else ''}{param_idx}{': ' if param_idx else ''}"
        log_message = super().format(record)
        return f"{self.COLORS.get(record.levelname, '')}{prefix}{log_message}{Style.RESET_ALL}"

def setup_logging(log_level: str = 'INFO', log_dir: Optional[str] = 'logs') -> logging.Logger:
    logger = logging.getLogger('RPropMLP')
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Queue handler for multiprocessing
    queue_handler = logging.handlers.QueueHandler(log_queue)
    time_format = "%H:%M"
    log_format = "%(levelname)s - %(asctime)s - %(message)s"
    queue_handler.setFormatter(ColoredFormatter(log_format, datefmt=time_format))
    logger.addHandler(queue_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter(log_format, datefmt=time_format))
    logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_dir:
        try:
            Path(log_dir).mkdir(exist_ok=True)
            file_handler = logging.FileHandler(
                filename=Path(log_dir) / 'training.log',
                mode='w',  # <--- overwrite every time
                encoding='utf-8'
            )
            file_handler.setLevel(logging.WARNING)
            file_handler.setFormatter(logging.Formatter(log_format, datefmt=time_format))
            logger.addHandler(file_handler)
        except (PermissionError, OSError) as e:
            logger.warning(f"Cannot write to log file in {log_dir}: {str(e)}. Logging to console only.")
    
    return logger