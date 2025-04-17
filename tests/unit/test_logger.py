"""
Unit tests for the Logger functionality.
"""

import pytest
from unittest.mock import patch, MagicMock
import io
import sys
import re
from datetime import datetime

from src.logger import get_logger, Logger, NullLogger, ConsoleLogger, LogLevel


class TestLogger:
    """Test suite for Logger classes."""

    @pytest.fixture
    def console_logger(self):
        """Get a console logger instance."""
        return ConsoleLogger()
    
    @pytest.fixture
    def null_logger(self):
        """Get a null logger instance."""
        return NullLogger()
    
    @pytest.fixture
    def captured_output(self):
        """Capture stdout for testing logger output."""
        # Create a StringIO object to capture output
        captured_output = io.StringIO()
        # Replace stdout with our capture object
        original_stdout = sys.stdout
        sys.stdout = captured_output
        # Provide the capture object
        yield captured_output
        # Restore stdout
        sys.stdout = original_stdout

    def test_get_logger(self):
        """Test the get_logger factory function."""
        logger = get_logger()
        assert isinstance(logger, Logger)
        assert isinstance(logger, ConsoleLogger)
    
    def test_null_logger_methods(self, null_logger):
        """Test that NullLogger implementations do nothing."""
        # None of these should produce any output or errors
        null_logger.debug("Test debug")
        null_logger.info("Test info")
        null_logger.warn("Test warning")
        null_logger.error("Test error")
        null_logger.set_level(LogLevel.DEBUG)
        
        # No assertions needed as we're testing the absence of behavior
    
    def test_console_logger_default_level(self, console_logger):
        """Test the default log level of ConsoleLogger."""
        assert console_logger.level == LogLevel.INFO
    
    def test_set_log_level(self, console_logger):
        """Test setting different log levels."""
        console_logger.set_level(LogLevel.DEBUG)
        assert console_logger.level == LogLevel.DEBUG
        
        console_logger.set_level(LogLevel.ERROR)
        assert console_logger.level == LogLevel.ERROR
    
    def test_log_level_filtering(self, monkeypatch, console_logger):
        """Test that messages are filtered based on log level."""
        outputs = []
        # Mock print to capture output
        monkeypatch.setattr('builtins.print', lambda *args, **kwargs: outputs.append(' '.join(str(a) for a in args)))
        
        # Set to INFO level
        console_logger.set_level(LogLevel.INFO)
        
        # DEBUG shouldn't log
        console_logger.debug("Debug message")
        assert not any("Debug message" in output for output in outputs)
        
        # INFO should log
        console_logger.info("Info message")
        assert any("Info message" in output for output in outputs)
        
        # Clear outputs
        outputs.clear()
        
        # Set to ERROR level
        console_logger.set_level(LogLevel.ERROR)
        
        # INFO shouldn't log at ERROR level
        console_logger.info("Another info")
        assert not any("Another info" in output for output in outputs)
        
        # ERROR should log
        console_logger.error("Error message")
        assert any("Error message" in output for output in outputs)
    
    def test_debug_log(self, monkeypatch):
        """Test debug level logging."""
        outputs = []
        # Mock print to capture output
        monkeypatch.setattr('builtins.print', lambda *args, **kwargs: outputs.append(' '.join(str(a) for a in args)))
        
        logger = ConsoleLogger()
        logger.set_level(LogLevel.DEBUG)
        logger.debug("Debug test message")
        
        # Check output contains the debug message
        assert any("DEBUG: Debug test message" in output for output in outputs)
    
    def test_info_log(self, monkeypatch):
        """Test info level logging."""
        outputs = []
        # Mock print to capture output
        monkeypatch.setattr('builtins.print', lambda *args, **kwargs: outputs.append(' '.join(str(a) for a in args)))
        
        logger = ConsoleLogger()
        logger.info("Info test message")
        
        # Check output contains the info message
        assert any("INFO: Info test message" in output for output in outputs)
    
    def test_warn_log(self, monkeypatch):
        """Test warning level logging."""
        outputs = []
        # Mock print to capture output
        monkeypatch.setattr('builtins.print', lambda *args, **kwargs: outputs.append(' '.join(str(a) for a in args)))
        
        logger = ConsoleLogger()
        logger.warn("Warning test message")
        
        # Check output contains the warning message
        assert any("WARNING: Warning test message" in output for output in outputs)
    
    def test_error_log(self, monkeypatch):
        """Test error level logging."""
        outputs = []
        # Mock print to capture output
        monkeypatch.setattr('builtins.print', lambda *args, **kwargs: outputs.append(' '.join(str(a) for a in args)))
        
        logger = ConsoleLogger()
        logger.error("Error test message")
        
        # Check output contains the error message
        assert any("ERROR: Error test message" in output for output in outputs)
    
    def test_log_with_payload(self, monkeypatch):
        """Test logging with a payload object."""
        outputs = []
        # Mock print to capture output
        monkeypatch.setattr('builtins.print', lambda *args, **kwargs: outputs.append(' '.join(str(a) for a in args)))
        
        logger = ConsoleLogger()
        payload = {"test": "data", "number": 123}
        logger.info("Message with payload", payload=payload)
        
        # Check output contains the message and payload
        assert any("INFO: Message with payload" in output for output in outputs)
        assert any(str(payload) in output for output in outputs)
    
    def test_log_with_context(self, monkeypatch):
        """Test logging with context information."""
        outputs = []
        # Mock print to capture output
        monkeypatch.setattr('builtins.print', lambda *args, **kwargs: outputs.append(' '.join(str(a) for a in args)))
        
        logger = ConsoleLogger()
        context = {"user_id": "123", "request_id": "abc"}
        logger.info("Message with context", context=context)
        
        # Check output contains the message and context
        assert any("INFO: Message with context" in output for output in outputs)
        assert any("user_id=123" in output for output in outputs)
        assert any("request_id=abc" in output for output in outputs)
    
    def test_error_with_exception_info(self, monkeypatch):
        """Test error logging with exception traceback."""
        outputs = []
        # Mock print to capture output
        monkeypatch.setattr('builtins.print', lambda *args, **kwargs: outputs.append(' '.join(str(a) for a in args)))
        
        logger = ConsoleLogger()
        try:
            # Generate an exception
            raise ValueError("Test exception")
        except ValueError:
            logger.error("Error with exception", exc_info=True)
            
        # Check output contains the error message
        assert any("ERROR: Error with exception" in output for output in outputs)
    
    def test_timestamp_format(self, monkeypatch):
        """Test that log messages include properly formatted timestamps."""
        outputs = []
        # Mock print to capture output
        monkeypatch.setattr('builtins.print', lambda *args, **kwargs: outputs.append(' '.join(str(a) for a in args)))
        
        logger = ConsoleLogger()
        logger.info("Timestamp test")
        
        # Join all outputs to a single string for easier testing
        log_output = '\n'.join(outputs)
        
        # Extract timestamp using regex
        timestamp_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})'
        match = re.search(timestamp_pattern, log_output)
        
        assert match is not None, "Timestamp pattern not found in log output"
        timestamp_str = match.group(1)
        
        # Verify timestamp format
        try:
            datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            pytest.fail("Timestamp is not in the expected format")
    
    def test_automatic_caller_context(self, monkeypatch):
        """Test that caller information is automatically added to context."""
        outputs = []
        # Mock print to capture output
        monkeypatch.setattr('builtins.print', lambda *args, **kwargs: outputs.append(' '.join(str(a) for a in args)))
        
        logger = ConsoleLogger()
        logger.info("Caller test")
        
        # Join all outputs to a single string for easier testing
        log_output = '\n'.join(outputs)
        
        # Check caller info is included
        assert "caller=" in log_output
        assert "test_logger.py:" in log_output
    
    def test_should_log_method(self, console_logger):
        """Test the internal _should_log method."""
        console_logger.set_level(LogLevel.INFO)
        
        # Direct testing of internal method
        assert console_logger._should_log(LogLevel.INFO) is True
        assert console_logger._should_log(LogLevel.DEBUG) is False
        assert console_logger._should_log(LogLevel.ERROR) is True
        
        console_logger.set_level(LogLevel.ERROR)
        assert console_logger._should_log(LogLevel.INFO) is False
        assert console_logger._should_log(LogLevel.ERROR) is True 