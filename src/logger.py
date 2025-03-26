from enum import Enum
from typing import Any, Optional


class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "debug"
    INFO = "info"
    WARN = "warning"
    ERROR = "error"


class Logger:
    """Base logger interface."""
    def debug(self, message: str, payload: Optional[Any] = None) -> None:
        """Log a debug message."""
        pass

    def info(self, message: str, payload: Optional[Any] = None) -> None:
        """Log an info message."""
        pass

    def warn(self, message: str, payload: Optional[Any] = None) -> None:
        """Log a warning message."""
        pass

    def error(self, message: str, payload: Optional[Any] = None) -> None:
        """Log an error message."""
        pass

    def set_level(self, level: LogLevel) -> None:
        """Set the log level."""
        pass


class NullLogger(Logger):
    """Logger implementation that does nothing."""
    pass


class ConsoleLogger(Logger):
    """Logger implementation that logs to the console."""
    def __init__(self) -> None:
        self.level = LogLevel.INFO

    def set_level(self, level: LogLevel) -> None:
        """Set the log level."""
        self.level = level

    def debug(self, message: str, payload: Optional[Any] = None) -> None:
        """Log a debug message."""
        if self._should_log(LogLevel.DEBUG):
            if payload:
                print(f"DEBUG: {message}", payload)
            else:
                print(f"DEBUG: {message}")

    def info(self, message: str, payload: Optional[Any] = None) -> None:
        """Log an info message."""
        if self._should_log(LogLevel.INFO):
            if payload:
                print(f"INFO: {message}", payload)
            else:
                print(f"INFO: {message}")

    def warn(self, message: str, payload: Optional[Any] = None) -> None:
        """Log a warning message."""
        if self._should_log(LogLevel.WARN):
            if payload:
                print(f"WARNING: {message}", payload)
            else:
                print(f"WARNING: {message}")

    def error(self, message: str, payload: Optional[Any] = None) -> None:
        """Log an error message."""
        if self._should_log(LogLevel.ERROR):
            if payload:
                print(f"ERROR: {message}", payload)
            else:
                print(f"ERROR: {message}")

    def _should_log(self, message_level: LogLevel) -> bool:
        """Check if a message should be logged based on the current log level."""
        levels = [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARN, LogLevel.ERROR]
        return levels.index(message_level) >= levels.index(self.level)


def get_logger() -> Logger:
    """Get a logger instance."""
    return ConsoleLogger() 