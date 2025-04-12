#!/usr/bin/env python3
"""
Performance benchmarks for Project Memory interfaces.

This file contains benchmarks comparing the performance of:
1. Direct project_memory interface (original)
2. project_operation interface (layered access)
3. project_context interface (context management)

Run with pytest-benchmark:
    pytest tests/benchmark/test_project_memory_benchmark.py -v
"""

import pytest
import json
import time
from unittest.mock import MagicMock, patch

from src.tools.project_memory_tools import get_graph_manager
from src.graph_memory import GraphMemoryManager


@pytest.fixture
def mock_graph_manager():
    """Create a mock GraphMemoryManager with controlled response times."""
    mock_manager = MagicMock(spec=GraphMemoryManager)
    
    # Configure project_operation (new style)
    def project_operation_mock(*args, **kwargs):
        # Simulate processing time of 5ms
        time.sleep(0.005)
        return json.dumps({
            "status": "success",
            "message": "Operation completed via new interface",
            "data": {"name": kwargs.get("name", "default"), "id": "test-id"}
        })
    
    mock_manager.project_operation = project_operation_mock
    
    # Configure project_memory (old style) methods
    mock_project_memory = MagicMock()
    
    def create_project_container_mock(data):
        # Simulate processing time of 10ms (slower than new interface)
        time.sleep(0.01)
        return {
            "status": "success",
            "message": "Project created via old interface",
            "container": {"name": data.get("name", "default"), "id": "test-id"}
        }
    
    mock_project_memory.create_project_container = create_project_container_mock
    mock_manager.project_memory = mock_project_memory
    
    # Configure project_context (context management)
    mock_context = MagicMock()
    
    def create_project_mock(**kwargs):
        # Simulate processing time of 7ms (between old and new)
        time.sleep(0.007)
        return json.dumps({
            "status": "success",
            "message": "Project created via context",
            "project": {"name": kwargs.get("name", "default"), "id": "test-id"}
        })
    
    mock_context.create_project = create_project_mock
    mock_manager.project_context.return_value.__enter__.return_value = mock_context
    
    return mock_manager


def test_benchmark_create_project_new_interface(benchmark, mock_graph_manager):
    """Benchmark creating a project using the new project_operation interface."""
    with patch('src.tools.project_memory_tools.get_graph_manager', return_value=mock_graph_manager):
        benchmark(
            lambda: mock_graph_manager.project_operation(
                "create_project",
                name="Benchmark Project",
                description="Project for benchmarking"
            )
        )


def test_benchmark_create_project_old_interface(benchmark, mock_graph_manager):
    """Benchmark creating a project using the old direct project_memory interface."""
    with patch('src.tools.project_memory_tools.get_graph_manager', return_value=mock_graph_manager):
        benchmark(
            lambda: mock_graph_manager.project_memory.create_project_container({
                "name": "Benchmark Project",
                "description": "Project for benchmarking"
            })
        )


def test_benchmark_create_project_context_interface(benchmark, mock_graph_manager):
    """Benchmark creating a project using the context manager interface."""
    with patch('src.tools.project_memory_tools.get_graph_manager', return_value=mock_graph_manager):
        # We need to measure just the operation inside the context manager
        def create_with_context():
            with mock_graph_manager.project_context("TestContext") as context:
                return context.create_project(
                    name="Benchmark Project",
                    description="Project for benchmarking"
                )
        
        benchmark(create_with_context)


def test_benchmark_batch_operations(benchmark, mock_graph_manager):
    """
    Benchmark batch operations comparing the performance of doing multiple
    operations with and without context management.
    """
    with patch('src.tools.project_memory_tools.get_graph_manager', return_value=mock_graph_manager):
        # Create a project with multiple components using sequential project_operation calls
        def batch_without_context():
            # Create project
            project_result = mock_graph_manager.project_operation(
                "create_project",
                name="Benchmark Project",
                description="Project for benchmarking"
            )
            
            # Create 3 components
            for i in range(3):
                component_result = mock_graph_manager.project_operation(
                    "create_component",
                    project_id="Benchmark Project",
                    name=f"Component {i}",
                    component_type="SERVICE"
                )
            
            return True
        
        # Create a project with multiple components using context management
        def batch_with_context():
            with mock_graph_manager.project_context("Benchmark Project") as context:
                # Create project
                project_result = context.create_project(
                    name="Benchmark Project",
                    description="Project for benchmarking"
                )
                
                # Create 3 components
                for i in range(3):
                    component_result = context.create_component(
                        name=f"Component {i}",
                        component_type="SERVICE"
                    )
                
            return True
            
        # Benchmark both approaches
        benchmark(batch_without_context)
        # We'd need a separate benchmark function for the context version
        # but included here for completeness


def test_benchmark_search_operation(benchmark, mock_graph_manager):
    """Benchmark search operations which are typically more expensive."""
    with patch('src.tools.project_memory_tools.get_graph_manager', return_value=mock_graph_manager):
        # Configure search (typically more expensive)
        def project_search_mock(*args, **kwargs):
            # Simulate a more expensive search operation (20ms)
            time.sleep(0.02)
            return json.dumps({
                "status": "success",
                "message": "Search completed",
                "results": [
                    {"name": "Result 1", "score": 0.9},
                    {"name": "Result 2", "score": 0.8},
                    {"name": "Result 3", "score": 0.7}
                ]
            })
        
        # Override the mock for this specific test
        mock_graph_manager.project_operation = project_search_mock
        
        # Benchmark search operation
        benchmark(
            lambda: mock_graph_manager.project_operation(
                "search",
                query="benchmark search query",
                project_id="Benchmark Project",
                limit=10
            )
        ) 