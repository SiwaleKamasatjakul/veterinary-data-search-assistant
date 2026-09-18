"""Execution-time logging.

Change from the original: the log file goes to the configured log directory
instead of a relative ``execution_time.log`` in the current working directory,
and the file handler is attached once rather than on every instantiation.
"""

from __future__ import annotations

import logging
import time

from tools.config_loader import ConfigManager


class MeasureTimePerformance:
    def __init__(self, log_file: str | None = None) -> None:
        self.start_time = 0.0
        self.end_time = 0.0
        self.execution_time = 0.0
        self.tokens = 0

        log_dir = ConfigManager.get_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = log_file or str(log_dir / "execution_time.log")

        self.logger = logging.getLogger("TimeLogger")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_file, encoding="utf-8")
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            )
            self.logger.addHandler(handler)

    def begin_process_time(self):
        self.tokens = 0
        self.start_time = time.time()
        return self

    def end_process_time(self):
        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time
        self.log_execution_time()
        return self

    def get_execution_time(self) -> float:
        return self.execution_time

    def get_tokens(self) -> int:
        return self.tokens

    def reset(self):
        self.start_time = self.end_time = self.execution_time = 0.0
        self.tokens = 0
        return self

    def token_count(self):
        self.tokens += 1
        return self

    def log_execution_time(self):
        self.logger.info(
            "Execution Time: %.6f seconds Tokens: %d", self.execution_time, self.tokens
        )
