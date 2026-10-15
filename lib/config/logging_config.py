import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging() -> None:
    log_directory = Path(__file__).resolve().parents[2] / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    if any(
        isinstance(handler, RotatingFileHandler)
        and Path(handler.baseFilename) == log_directory / "response_api.log"
        for handler in root_logger.handlers
    ):
        return

    file_handler = RotatingFileHandler(
        log_directory / "response_api.log",
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
        )
    )
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)
