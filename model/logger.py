import sys
import logging

COLORS = {
    "DEBUG": "\033[90m",  # Grey
    "INFO": "\033[94m",   # Blue
    "WARNING": "\033[93m",  # Yellow
    "ERROR": "\033[91m",    # Red
    "CRITICAL": "\033[95m",  # Magenta
}
RESET = "\033[0m"


class ColorLogger(logging.Formatter):
    def format(self, record) -> str:
        color = COLORS.get(record.levelname, "")
        message = super().format(record)
        return f"{color}{message}{RESET}"

 
    
def setup_logger(
        level = logging.INFO,
        log_file : str = None,
) -> logging.Logger:
    logger = logging.getLogger("stream_logger")
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    fmt = "[%(asctime)s] [%(levelname)s] %(message)s"
    datefmt = "%H:%M:%S.%f"
    handler.setFormatter(ColorLogger(fmt, datefmt))
    
    logger.addHandler(handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
    
logger = setup_logger()
    