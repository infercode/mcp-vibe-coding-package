import unittest
import time
import statistics
from unittest.mock import patch, MagicMock

from src.graph_memory.base_manager import BaseManager
from src.graph_memory.entity_manager import EntityManager
from src.graph_memory.relation_manager import RelationManager
from src.graph_memory.observation_manager import ObservationManager
from src.graph_memory.search_manager import SearchManager
from src.graph_memory.embedding_adapter import EmbeddingAdapter

class TestPerformanceBenchmark(unittest.TestCase):
    """Performance benchmarking tests for the graph memory system."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a mock logger
        self.mock_logger = MagicMock()
        
        # Set up real components with mocked Neo4j connection
        with patch('neo4j.GraphDatabase.driver') as self.mock_driver:
            # Set up mock driver instance
            self.mock_driver_instance = MagicMock()
            self.mock_driver.return_value = self.mock_driver_instance
            
            # Create real components
            self.base_manager = BaseManager(self.mock_logger)
            self.base_manager.initialize()
            
            # Create dependent components
            self.entity_manager = EntityManager(self.base_manager)
            self.relation_manager = RelationManager(self.base_manager)
            self.observation_manager = ObservationManager(self.base_manager)
            
            # Create embedding adapter with mock
            self.mock_embedding_adapter = MagicMock(spec=EmbeddingAdapter)
            self.search_manager = SearchManager(self.base_manager, self.mock_embedding_adapter)
        
        # Sample entity batch for testing
        self.sample_entities = [
            {
                "name": f"benchmark_entity_{i}",
                "entityType": "BenchmarkType",
                "description": f"Entity for benchmarking operations {i}",
                "tags": ["benchmark", "test", f"tag_{i}"]
            }
            for i in range(100)  # Create 100 sample entities
        ]
        
        # Configure mock responses
        # For entity creation
        self.base_manager.execute_query = MagicMock()
        self.base_manager.execute_query.return_value = [
            {"e": {"id": i + 1000, "properties": entity}} 
            for i, entity in enumerate(self.sample_entities)
        ]
        
        # For embedding generation
        self.mock_embedding_adapter.generate_embedding.return_value = [0.1, 0.2, 0.3]
        
        # Performance metrics storage
        self.metrics = {}
    
    def tearDown(self):
        """Clean up after each test."""
        self.base_manager.close()
        
        # Print metrics summary if any
        if self.metrics:
            print("\nPerformance Metrics Summary:")
            for operation, times in self.metrics.items():
                print(f"  {operation}:")
                print(f"    Min: {min(times):.6f}s")
                print(f"    Max: {max(times):.6f}s")
                print(f"    Avg: {statistics.mean(times):.6f}s")
                print(f"    Med: {statistics.median(times):.6f}s")
                if len(times) > 1:
                    print(f"    Std: {statistics.stdev(times):.6f}s")
    
    def measure_performance(self, operation_name, operation_func, iterations=10):
        """Measure performance of an operation over multiple iterations."""
        times = []
        
        for _ in range(iterations):
            start_time = time.time()
            result = operation_func()
            end_time = time.time()
            times.append(end_time - start_time)
        
        self.metrics[operation_name] = times
        return statistics.mean(times)
    
    def test_entity_creation_performance(self):
        """Benchmark entity creation performance."""
        # Measure single entity creation
        avg_time = self.measure_performance(
            "Single Entity Creation",
            lambda: self.entity_manager.create_entity(self.sample_entities[0])
        )
        
        # Assert performance is within expected range
        # This is just a placeholder assertion since we're mocking the database
        self.assertLess(avg_time, 0.1, "Single entity creation took too long")
        
        # Measure batch entity creation
        batch_avg_time = self.measure_performance(
            "Batch Entity Creation (100 entities)",
            lambda: self.entity_manager.create_entities(self.sample_entities),
            iterations=5  # Fewer iterations for batch operation
        )
        
        # Assert batch creation is more efficient per entity
        # Since we're just measuring code performance, not real DB performance
        self.assertLess(
            batch_avg_time / len(self.sample_entities),
            avg_time,
            "Batch creation should be more efficient per entity"
        )
    
    def test_entity_retrieval_performance(self):
        """Benchmark entity retrieval performance."""
        # Measure entity retrieval by name
        avg_time = self.measure_performance(
            "Entity Retrieval by Name",
            lambda: self.entity_manager.get_entity("benchmark_entity_0")
        )
        
        # Assert performance is within expected range
        self.assertLess(avg_time, 0.05, "Entity retrieval took too long")
    
    def test_relationship_operations_performance(self):
        """Benchmark relationship operation performance."""
        # Measure relationship creation
        relation_avg_time = self.measure_performance(
            "Relationship Creation",
            lambda: self.relation_manager.create_relationship(
                source_entity="benchmark_entity_0",
                target_entity="benchmark_entity_1",
                relation_type="BENCHMARK_RELATION",
                properties={"weight": 0.8}
            )
        )
        
        # Assert performance is within expected range
        self.assertLess(relation_avg_time, 0.1, "Relationship creation took too long")
        
        # Measure relationship retrieval
        retrieval_avg_time = self.measure_performance(
            "Relationship Retrieval",
            lambda: self.relation_manager.get_relationships(entity_name="benchmark_entity_0")
        )
        
        # Assert performance is within expected range
        self.assertLess(retrieval_avg_time, 0.05, "Relationship retrieval took too long")
    
    def test_search_performance(self):
        """Benchmark search operation performance."""
        # Measure text search
        text_search_avg_time = self.measure_performance(
            "Text Search",
            lambda: self.search_manager.search_by_text("benchmark")
        )
        
        # Assert performance is within expected range
        self.assertLess(text_search_avg_time, 0.1, "Text search took too long")
        
        # Measure semantic search
        self.base_manager.execute_query.return_value = [
            {"entity": {"id": i + 1000, "properties": entity}, "score": 0.9 - (i * 0.01)}
            for i, entity in enumerate(self.sample_entities[:10])
        ]
        
        semantic_search_avg_time = self.measure_performance(
            "Semantic Search",
            lambda: self.search_manager.semantic_search("benchmark entity for operations")
        )
        
        # Assert performance is within expected range (should be slightly slower due to embedding)
        self.assertLess(semantic_search_avg_time, 0.15, "Semantic search took too long")
    
    def test_compare_refactored_vs_original(self):
        """Compare performance between refactored and original implementation (simulated)."""
        # This is a placeholder test that simulates comparison
        # In a real scenario, you would have benchmarks for both implementations
        
        # Simulate original performance
        original_performance = {
            "entity_creation": 0.05,  # seconds per operation
            "relation_creation": 0.07,
            "text_search": 0.08,
            "semantic_search": 0.12
        }
        
        # Measure refactored performance
        refactored_performance = {
            "entity_creation": self.measure_performance(
                "Entity Creation (Comparison)",
                lambda: self.entity_manager.create_entity(self.sample_entities[0])
            ),
            "relation_creation": self.measure_performance(
                "Relation Creation (Comparison)",
                lambda: self.relation_manager.create_relationship(
                    source_entity="benchmark_entity_0",
                    target_entity="benchmark_entity_1",
                    relation_type="BENCHMARK_RELATION"
                )
            ),
            "text_search": self.measure_performance(
                "Text Search (Comparison)",
                lambda: self.search_manager.search_by_text("benchmark")
            ),
            "semantic_search": self.measure_performance(
                "Semantic Search (Comparison)",
                lambda: self.search_manager.semantic_search("benchmark entity")
            ),
        }
        
        # Print comparison
        print("\nPerformance Comparison (Refactored vs Original):")
        for operation, original_time in original_performance.items():
            refactored_time = refactored_performance[operation]
            improvement = ((original_time - refactored_time) / original_time) * 100
            print(f"  {operation}: {improvement:.2f}% improvement")
            
            # We don't assert here since this is simulated data
            # In reality, you would compare actual measurements
    
    def test_memory_usage(self):
        """Test memory usage of the system under load."""
        # This is a placeholder for memory profiling
        # In a real system, you would use a memory profiler like memory_profiler
        # and tracemalloc to measure actual memory usage
        
        try:
            import tracemalloc
            
            # Start memory tracking
            tracemalloc.start()
            
            # Create a bunch of entities and relationships to simulate load
            for i in range(10):  # Reduced for testing
                self.entity_manager.create_entity(self.sample_entities[i])
                if i > 0:
                    self.relation_manager.create_relationship(
                        source_entity=f"benchmark_entity_{i}",
                        target_entity=f"benchmark_entity_{i-1}",
                        relation_type="NEXT"
                    )
            
            # Get memory snapshot
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            print(f"\nMemory Usage:")
            print(f"  Current: {current / 1024 / 1024:.2f} MB")
            print(f"  Peak: {peak / 1024 / 1024:.2f} MB")
            
            # We don't assert here since memory usage can vary by environment
        except ImportError:
            print("\nTracemalloc not available, skipping memory usage test")

if __name__ == '__main__':
    unittest.main() 