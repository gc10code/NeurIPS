from enum import Enum
from datetime import datetime
from colorama import Fore, Style, init
from typing import Any

# Inizializza colorama
init(autoreset=True)
DEBUG_MODE = False
TIME_MODE = False

class LoggingLevels(Enum):
    DEBUG  = (Fore.WHITE, Style.DIM,     "DEBUG")
    INFO   = (Fore.WHITE, Style.NORMAL,  "INFO")
    REPORT = (Fore.BLUE,  Style.NORMAL,  "REPORT")
    SUCCESS= (Fore.GREEN, Style.NORMAL,  "SUCCESS")
    WARN   = (Fore.YELLOW,Style.NORMAL,  "WARN")
    ERROR  = (Fore.RED,   Style.NORMAL,  "ERROR")
    EXIT = (Fore.MAGENTA,   Style.BRIGHT,  "EXIT")

def lprint(level: LoggingLevels = LoggingLevels.INFO, text: str = "", timestamp=True) -> None:
    # System Configs
    if not TIME_MODE:
        timestamp = False
    if level == LoggingLevels.DEBUG and not DEBUG_MODE:
        return
    
    fore_color, style, label = level.value
    
    time_str = ""
    if timestamp:
        time_str = datetime.now().strftime("%H:%M") + " "
    
    print(f"{fore_color}{style}{time_str}{label} - {text}{Style.RESET_ALL}")
