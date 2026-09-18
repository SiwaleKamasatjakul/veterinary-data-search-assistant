"""Rotating file + console logger, anchored to the configured log directory."""

import logging
from logging.handlers import RotatingFileHandler

from tools.config_loader import ConfigManager


class CustomLogger:
    def __init__(
        self,
        name,
        level=logging.INFO,
        max_log_size=5 * 1024 * 1024,
        backup_count=5,
        enable_console=True,
    ):
        log_file_path = ConfigManager.get_log_file_path()
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            self.logger.setLevel(level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

            file_handler = RotatingFileHandler(
                log_file_path, maxBytes=max_log_size, backupCount=backup_count, encoding="utf-8"
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            if enable_console:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(level)
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)

    def get_logger(self) -> logging.Logger:
        return self.logger
