"""
Benchmark utilities for measuring performance of lesson memory operations.

This module provides tools to measure the performance of both direct lesson memory
operations and operations through the GraphMemoryManager interface.
"""

import time
import json
import statistics
from typing import Callable, Dict, List, Any, Optional, Tuple

from src.graph_memory import GraphMemoryManager
from src.lesson_memory import LessonMemoryManager


class PerformanceBenchmark:
    """
    Utility class for benchmarking memory operations performance.
    
    This class provides methods to compare performance between direct LessonMemoryManager
    operations and equivalent operations through GraphMemoryManager's layered interface.
    """
    
    def __init__(self, graph_manager: GraphMemoryManager):
        """
        Initialize the benchmark utility.
        
        Args:
            graph_manager: An initialized GraphMemoryManager instance
        """
        self.graph_manager = graph_manager
        self.lesson_memory = graph_manager.lesson_memory
        self.results = {}
    
    def benchmark_operation(
        self, 
        operation_name: str,
        direct_func: Callable,
        direct_args: List[Any],
        direct_kwargs: Dict[str, Any],
        layered_func: Callable,
        layered_args: List[Any],
        layered_kwargs: Dict[str, Any],
        iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark a specific operation comparing direct vs layered interface.
        
        Args:
            operation_name: Name of the operation being benchmarked
            direct_func: Function reference for direct LessonMemoryManager method
            direct_args: List of positional arguments for direct method
            direct_kwargs: Dict of keyword arguments for direct method
            layered_func: Function reference for layered interface method
            layered_args: List of positional arguments for layered method
            layered_kwargs: Dict of keyword arguments for layered method
            iterations: Number of iterations to run for each method
        
        Returns:
            Dict containing benchmark results including timing statistics
        """
        print(f"Benchmarking {operation_name}...")
        
        # Measure direct method performance
        direct_times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            direct_func(*direct_args, **direct_kwargs)
            end_time = time.perf_counter()
            direct_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Measure layered method performance
        layered_times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            layered_func(*layered_args, **layered_kwargs)
            end_time = time.perf_counter()
            layered_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        result = {
            "operation": operation_name,
            "iterations": iterations,
            "direct": {
                "mean": statistics.mean(direct_times),
                "median": statistics.median(direct_times),
                "min": min(direct_times),
                "max": max(direct_times),
                "stdev": statistics.stdev(direct_times) if len(direct_times) > 1 else 0
            },
            "layered": {
                "mean": statistics.mean(layered_times),
                "median": statistics.median(layered_times),
                "min": min(layered_times),
                "max": max(layered_times),
                "stdev": statistics.stdev(layered_times) if len(layered_times) > 1 else 0
            }
        }
        
        # Calculate overhead
        result["overhead"] = {
            "mean_pct": ((result["layered"]["mean"] / result["direct"]["mean"]) - 1) * 100,
            "median_pct": ((result["layered"]["median"] / result["direct"]["median"]) - 1) * 100
        }
        
        self.results[operation_name] = result
        return result
    
    def benchmark_create_operations(self, container_name: str, iterations: int = 100) -> Dict[str, Any]:
        """
        Benchmark lesson creation operations.
        
        Args:
            container_name: Name of the container to use for testing
            iterations: Number of iterations to run
            
        Returns:
            Dict containing benchmark results
        """
        direct_func = self.lesson_memory.create_lesson_entity
        direct_args = [container_name, "BenchmarkLesson", "BestPractice", None, None]
        direct_kwargs = {}
        
        layered_func = self.graph_manager.lesson_operation
        layered_args = ["create"]
        layered_kwargs = {
            "container_name": container_name,
            "name": "BenchmarkLesson",
            "lesson_type": "BestPractice"
        }
        
        return self.benchmark_operation(
            "create_lesson",
            direct_func, direct_args, direct_kwargs,
            layered_func, layered_args, layered_kwargs,
            iterations
        )
    
    def benchmark_observe_operations(self, container_name: str, entity_name: str, iterations: int = 100) -> Dict[str, Any]:
        """
        Benchmark observation creation operations.
        
        Args:
            container_name: Name of the container to use for testing
            entity_name: Name of the entity to create observations for
            iterations: Number of iterations to run
            
        Returns:
            Dict containing benchmark results
        """
        direct_func = self.lesson_memory.create_structured_lesson_observations
        direct_kwargs = {
            "entity_name": entity_name,
            "container_name": container_name,
            "what_was_learned": "Benchmark observation",
            "why_it_matters": "Performance testing",
            "confidence": 0.9
        }
        direct_args = []
        
        layered_func = self.graph_manager.lesson_operation
        layered_args = ["observe"]
        layered_kwargs = {
            "entity_name": entity_name,
            "container_name": container_name,
            "what_was_learned": "Benchmark observation",
            "why_it_matters": "Performance testing",
            "confidence": 0.9
        }
        
        return self.benchmark_operation(
            "create_observation",
            direct_func, direct_args, direct_kwargs,
            layered_func, layered_args, layered_kwargs,
            iterations
        )
    
    def benchmark_search_operations(self, query: str, iterations: int = 50) -> Dict[str, Any]:
        """
        Benchmark search operations.
        
        Args:
            query: Search query to use
            iterations: Number of iterations to run
            
        Returns:
            Dict containing benchmark results
        """
        direct_func = self.lesson_memory.search_lesson_semantic
        direct_args = []
        direct_kwargs = {"query": query, "limit": 10}
        
        layered_func = self.graph_manager.lesson_operation
        layered_args = ["search"]
        layered_kwargs = {"query": query, "limit": 10}
        
        return self.benchmark_operation(
            "search_lessons",
            direct_func, direct_args, direct_kwargs,
            layered_func, layered_args, layered_kwargs,
            iterations
        )
    
    def benchmark_context_operations(self, project_name: str, container_name: str, iterations: int = 50) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Benchmark context-based operations.
        
        Args:
            project_name: Project name to use
            container_name: Container name to use
            iterations: Number of iterations to run
            
        Returns:
            Tuple of benchmark results for context creation and operations
        """
        # First benchmark context creation
        direct_func = lambda p, c: (p, c)  # Just a placeholder, no direct equivalent
        direct_args = [project_name, container_name]
        direct_kwargs = {}
        
        layered_func = self.graph_manager.lesson_context
        layered_args = [project_name, container_name]
        layered_kwargs = {}
        
        context_result = self.benchmark_operation(
            "context_creation",
            direct_func, direct_args, direct_kwargs,
            layered_func, layered_args, layered_kwargs,
            iterations
        )
        
        # Then benchmark operations within context
        def direct_operation():
            return self.lesson_memory.create_lesson_entity(
                container_name, "ContextBenchmark", "BestPractice", None, None
            )
        
        def layered_operation():
            with self.graph_manager.lesson_context(project_name, container_name) as ctx:
                return ctx.create("ContextBenchmark", "BestPractice")
        
        direct_func = direct_operation
        direct_args = []
        direct_kwargs = {}
        
        layered_func = layered_operation
        layered_args = []
        layered_kwargs = {}
        
        operation_result = self.benchmark_operation(
            "context_operations",
            direct_func, direct_args, direct_kwargs,
            layered_func, layered_args, layered_kwargs,
            iterations
        )
        
        return context_result, operation_result
    
    def run_all_benchmarks(self, 
                          container_name: str = "BenchmarkContainer", 
                          entity_name: str = "BenchmarkEntity",
                          iterations: int = 50) -> Dict[str, Any]:
        """
        Run all benchmarks and collect results.
        
        Args:
            container_name: Container name to use for testing
            entity_name: Entity name to use for testing
            iterations: Number of iterations to run for each test
            
        Returns:
            Dict containing all benchmark results
        """
        # Create container and entity if they don't exist
        try:
            self.graph_manager.lesson_operation(
                "create",
                container_name=container_name,
                name=entity_name,
                lesson_type="BestPractice"
            )
        except Exception:
            # Entity might already exist, continue
            pass
        
        # Run all benchmarks
        self.benchmark_create_operations(container_name, iterations)
        self.benchmark_observe_operations(container_name, entity_name, iterations) 
        self.benchmark_search_operations("benchmark test", iterations)
        self.benchmark_context_operations("BenchmarkProject", container_name, iterations)
        
        # Generate summary
        summary = {
            "total_operations_benchmarked": len(self.results),
            "average_overhead_pct": statistics.mean([
                r["overhead"]["mean_pct"] for r in self.results.values()
            ]),
            "max_overhead_pct": max([
                r["overhead"]["mean_pct"] for r in self.results.values()
            ]),
            "operations": list(self.results.keys())
        }
        
        self.results["summary"] = summary
        return self.results
    
    def print_results(self, detailed: bool = False) -> None:
        """
        Print benchmark results in a formatted way.
        
        Args:
            detailed: Whether to include detailed stats for each operation
        """
        if not self.results:
            print("No benchmark results available. Run benchmarks first.")
            return
        
        print("\n=== BENCHMARK RESULTS ===\n")
        
        if "summary" in self.results:
            summary = self.results["summary"]
            print(f"Operations benchmarked: {summary['total_operations_benchmarked']}")
            print(f"Average overhead: {summary['average_overhead_pct']:.2f}%")
            print(f"Maximum overhead: {summary['max_overhead_pct']:.2f}%")
            print(f"Operations: {', '.join(summary['operations'])}")
            print("\n")
        
        for op_name, result in self.results.items():
            if op_name == "summary":
                continue
                
            print(f"Operation: {op_name}")
            print(f"  Direct API median: {result['direct']['median']:.3f}ms")
            print(f"  Layered API median: {result['layered']['median']:.3f}ms")
            print(f"  Overhead: {result['overhead']['median_pct']:.2f}%")
            
            if detailed:
                print("  Details:")
                print(f"    Direct mean: {result['direct']['mean']:.3f}ms (stdev: {result['direct']['stdev']:.3f}ms)")
                print(f"    Layered mean: {result['layered']['mean']:.3f}ms (stdev: {result['layered']['stdev']:.3f}ms)")
                print(f"    Direct range: {result['direct']['min']:.3f}ms - {result['direct']['max']:.3f}ms")
                print(f"    Layered range: {result['layered']['min']:.3f}ms - {result['layered']['max']:.3f}ms")
            
            print("")
    
    def save_results(self, filename: str) -> None:
        """
        Save benchmark results to a JSON file.
        
        Args:
            filename: Path to the file where results should be saved
        """
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filename}")


def run_benchmark(graph_manager: GraphMemoryManager,
                 container_name: str = "BenchmarkContainer",
                 iterations: int = 50,
                 save_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to run all benchmarks and return results.
    
    Args:
        graph_manager: Initialized GraphMemoryManager instance
        container_name: Container name to use for testing
        iterations: Number of iterations for each benchmark
        save_file: If provided, save results to this file
        
    Returns:
        Dict containing benchmark results
    """
    benchmark = PerformanceBenchmark(graph_manager)
    results = benchmark.run_all_benchmarks(container_name, iterations=iterations)
    benchmark.print_results()
    
    if save_file:
        benchmark.save_results(save_file)
    
    return results 