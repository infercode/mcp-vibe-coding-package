from enum import Enum
import time
import datetime
import inspect
import os
import traceback
from typing import Any, Dict, Optional


class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "debug"
    INFO = "info"
    WARN = "warning"
    ERROR = "error"


class Logger:
    """Base logger interface."""
    def debug(self, message: str, payload: Optional[Any] = None, context: Optional[Dict[str, Any]] = None) -> None:
        """Log a debug message."""
        pass

    def info(self, message: str, payload: Optional[Any] = None, context: Optional[Dict[str, Any]] = None) -> None:
        """Log an info message."""
        pass

    def warn(self, message: str, payload: Optional[Any] = None, context: Optional[Dict[str, Any]] = None) -> None:
        """Log a warning message."""
        pass

    def error(self, message: str, payload: Optional[Any] = None, context: Optional[Dict[str, Any]] = None, exc_info: bool = False) -> None:
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

    def debug(self, message: str, payload: Optional[Any] = None, context: Optional[Dict[str, Any]] = None) -> None:
        """Log a debug message."""
        if self._should_log(LogLevel.DEBUG):
            context = context or {}
            self._log("DEBUG", message, payload, context)

    def info(self, message: str, payload: Optional[Any] = None, context: Optional[Dict[str, Any]] = None) -> None:
        """Log an info message."""
        if self._should_log(LogLevel.INFO):
            context = context or {}
            self._log("INFO", message, payload, context)

    def warn(self, message: str, payload: Optional[Any] = None, context: Optional[Dict[str, Any]] = None) -> None:
        """Log a warning message."""
        if self._should_log(LogLevel.WARN):
            context = context or {}
            self._log("WARNING", message, payload, context)

    def error(self, message: str, payload: Optional[Any] = None, context: Optional[Dict[str, Any]] = None, exc_info: bool = False) -> None:
        """Log an error message."""
        if self._should_log(LogLevel.ERROR):
            context = context or {}
            self._log("ERROR", message, payload, context)
            if exc_info:
                print(traceback.format_exc())

    def _should_log(self, message_level: LogLevel) -> bool:
        """Check if a message should be logged based on the current log level."""
        levels = [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARN, LogLevel.ERROR]
        return levels.index(message_level) >= levels.index(self.level)

    def _log(self, level: str, message: str, payload: Optional[Any] = None, context: Optional[Dict[str, Any]] = None) -> None:
        """Internal method to format and print log messages."""
        # Get timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Initialize context if None
        ctx = context or {}
        
        # Get caller information if not in context
        if "caller" not in ctx:
            frame = inspect.currentframe()
            if frame:
                try:
                    caller_frame = frame.f_back
                    if caller_frame and caller_frame.f_back:
                        caller_frame = caller_frame.f_back  # Skip _log and the actual log method
                        filename = os.path.basename(caller_frame.f_code.co_filename)
                        lineno = caller_frame.f_lineno
                        function = caller_frame.f_code.co_name
                        ctx["caller"] = f"{filename}:{lineno} ({function})"
                finally:
                    del frame  # Avoid reference cycles
        
        # Format context string
        context_str = ""
        if ctx:
            context_items = [f"{k}={v}" for k, v in ctx.items()]
            context_str = f" [{', '.join(context_items)}]"
        
        # Format and print log message
        if payload:
            print(f"{timestamp} {level}: {message}{context_str}", payload)
        else:
            print(f"{timestamp} {level}: {message}{context_str}")


def get_logger() -> Logger:
    """Get a logger instance."""
    return ConsoleLogger() 