import unittest
from unittest.mock import patch, MagicMock

from src.graph_memory.search_manager import SearchManager
from src.graph_memory.base_manager import BaseManager
from src.graph_memory.embedding_adapter import EmbeddingAdapter

class TestSearchManager(unittest.TestCase):
    """Test suite for SearchManager class responsible for searching entities in the graph."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a mock base manager
        self.mock_base_manager = MagicMock(spec=BaseManager)
        
        # Create a mock embedding adapter
        self.mock_embedding_adapter = MagicMock(spec=EmbeddingAdapter)
        
        # Create search manager with mock dependencies
        self.search_manager = SearchManager(self.mock_base_manager, self.mock_embedding_adapter)
        
        # Sample entity results for testing
        self.sample_entities = [
            {
                "id": 123,
                "name": "test_entity_1",
                "entityType": "TestType",
                "description": "A test entity with specific features",
                "tags": ["test", "search"]
            },
            {
                "id": 456,
                "name": "test_entity_2",
                "entityType": "TestType",
                "description": "Another test entity with different features",
                "tags": ["test", "feature"]
            }
        ]
    
    def test_search_entities_by_text(self):
        """Test searching entities by text content."""
        # Setup mock response
        self.mock_base_manager.execute_query.return_value = [
            {"e": {"id": 123, "properties": self.sample_entities[0]}}
        ]
        
        # Call method under test
        results = self.search_manager.search_by_text("specific features")
        
        # Assertions
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].get("name"), "test_entity_1")
        self.assertEqual(results[0].get("id"), 123)
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
        # Verify that the query contains text search terms
        call_args = self.mock_base_manager.execute_query.call_args[0]
        self.assertTrue("CONTAINS" in call_args[0] or "specific features" in call_args[0])
    
    def test_search_entities_by_property(self):
        """Test searching entities by exact property values."""
        # Setup mock response
        self.mock_base_manager.execute_query.return_value = [
            {"e": {"id": 123, "properties": self.sample_entities[0]}}
        ]
        
        # Call method under test
        results = self.search_manager.search_by_property(
            property_name="entityType",
            property_value="TestType",
            entity_type=None
        )
        
        # Assertions
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].get("entityType"), "TestType")
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
        # Verify that the query contains property matching
        call_args = self.mock_base_manager.execute_query.call_args[0]
        self.assertTrue("entityType" in call_args[0] and "TestType" in call_args[0])
    
    def test_search_entities_by_tag(self):
        """Test searching entities by tag."""
        # Setup mock response
        self.mock_base_manager.execute_query.return_value = [
            {"e": {"id": 456, "properties": self.sample_entities[1]}}
        ]
        
        # Call method under test
        results = self.search_manager.search_by_tag("feature")
        
        # Assertions
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].get("name"), "test_entity_2")
        self.assertTrue("feature" in results[0].get("tags"))
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
        # Verify that the query contains tag matching
        call_args = self.mock_base_manager.execute_query.call_args[0]
        self.assertTrue("tags" in call_args[0] and "feature" in call_args[0])
    
    def test_semantic_search(self):
        """Test semantic search using embeddings."""
        # Setup mock responses
        # Mock embedding generation
        self.mock_embedding_adapter.generate_embedding.return_value = [0.1, 0.2, 0.3]
        
        # Mock vector search query response
        self.mock_base_manager.execute_query.return_value = [
            {
                "entity": {"id": 123, "properties": self.sample_entities[0]},
                "score": 0.85
            },
            {
                "entity": {"id": 456, "properties": self.sample_entities[1]},
                "score": 0.72
            }
        ]
        
        # Call method under test
        results = self.search_manager.semantic_search(
            query_text="entity with features",
            limit=5,
            threshold=0.7
        )
        
        # Assertions
        self.assertEqual(len(results), 2)
        # Results should be sorted by score (highest first)
        self.assertEqual(results[0].get("id"), 123)
        self.assertEqual(results[0].get("score"), 0.85)
        self.assertEqual(results[1].get("id"), 456)
        self.assertEqual(results[1].get("score"), 0.72)
        
        # Verify embedding generation was called
        self.mock_embedding_adapter.generate_embedding.assert_called_once_with("entity with features")
        
        # Verify query execution with vector search
        self.mock_base_manager.execute_query.assert_called_once()
        # Verify that the query contains vector search
        call_args = self.mock_base_manager.execute_query.call_args[0]
        self.assertTrue("RETURN" in call_args[0] and "score" in call_args[0])
    
    def test_combined_search(self):
        """Test combined search with both text and semantic components."""
        # Setup mock responses
        # Mock embedding generation
        self.mock_embedding_adapter.generate_embedding.return_value = [0.1, 0.2, 0.3]
        
        # Mock combined search query response
        self.mock_base_manager.execute_query.return_value = [
            {
                "entity": {"id": 123, "properties": self.sample_entities[0]},
                "text_score": 0.9,
                "semantic_score": 0.85,
                "combined_score": 0.875
            }
        ]
        
        # Call method under test
        results = self.search_manager.combined_search(
            text_query="specific",
            semantic_query="entity with features",
            entity_type="TestType",
            text_weight=0.5,
            semantic_weight=0.5,
            limit=5
        )
        
        # Assertions
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].get("id"), 123)
        self.assertEqual(results[0].get("combined_score"), 0.875)
        
        # Verify embedding generation was called
        self.mock_embedding_adapter.generate_embedding.assert_called_once_with("entity with features")
        
        # Verify query execution with combined search parameters
        self.mock_base_manager.execute_query.assert_called_once()
        # Verify that the query contains both text and semantic search elements
        call_args = self.mock_base_manager.execute_query.call_args[0]
        self.assertTrue("TestType" in call_args[0] and "specific" in call_args[0])
    
    def test_advanced_filtering(self):
        """Test advanced entity filtering."""
        # Setup mock response
        self.mock_base_manager.execute_query.return_value = [
            {"e": {"id": 123, "properties": self.sample_entities[0]}}
        ]
        
        # Create complex filter criteria
        filter_criteria = {
            "entityType": "TestType",
            "tags": ["test", "search"],
            "property_conditions": [
                {"property": "description", "operator": "CONTAINS", "value": "specific"}
            ]
        }
        
        # Call method under test
        results = self.search_manager.advanced_filter(filter_criteria)
        
        # Assertions
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].get("name"), "test_entity_1")
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
        # Verify that the query contains complex filtering
        call_args = self.mock_base_manager.execute_query.call_args[0]
        self.assertTrue("WHERE" in call_args[0] and "test" in call_args[0] and "specific" in call_args[0])
    
    def test_search_with_relevance_sorting(self):
        """Test search with relevance-based sorting."""
        # Setup mock response
        mock_results_with_relevance = [
            {
                "entity": {"id": 456, "properties": self.sample_entities[1]},
                "relevance_score": 0.95
            },
            {
                "entity": {"id": 123, "properties": self.sample_entities[0]},
                "relevance_score": 0.85
            }
        ]
        self.mock_base_manager.execute_query.return_value = mock_results_with_relevance
        
        # Call method under test
        results = self.search_manager.search_with_relevance("test entity")
        
        # Assertions
        self.assertEqual(len(results), 2)
        # Results should be sorted by relevance score (highest first)
        self.assertEqual(results[0].get("id"), 456)
        self.assertEqual(results[0].get("relevance_score"), 0.95)
        self.assertEqual(results[1].get("id"), 123)
        self.assertEqual(results[1].get("relevance_score"), 0.85)
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
        # Verify that the query includes relevance calculation
        call_args = self.mock_base_manager.execute_query.call_args[0]
        self.assertTrue("relevance" in call_args[0].lower() or "score" in call_args[0].lower())
    
    def test_search_with_pagination(self):
        """Test search with pagination support."""
        # Setup mock response for first page
        self.mock_base_manager.execute_query.return_value = [
            {"e": {"id": 123, "properties": self.sample_entities[0]}}
        ]
        
        # Call method under test for first page
        results, has_more = self.search_manager.paginated_search(
            search_term="test entity",
            page=1,
            page_size=1
        )
        
        # Assertions for first page
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].get("id"), 123)
        self.assertTrue(has_more)
        
        # Verify query execution with pagination parameters
        self.mock_base_manager.execute_query.assert_called_once()
        # Verify that the query includes pagination
        call_args = self.mock_base_manager.execute_query.call_args[0]
        self.assertTrue("SKIP" in call_args[0] and "LIMIT" in call_args[0])
        
        # Reset mock for second page test
        self.mock_base_manager.execute_query.reset_mock()
        
        # Setup mock response for second page (empty, indicating no more results)
        self.mock_base_manager.execute_query.return_value = []
        
        # Call method under test for second page
        results, has_more = self.search_manager.paginated_search(
            search_term="test entity",
            page=2,
            page_size=1
        )
        
        # Assertions for second page
        self.assertEqual(len(results), 0)
        self.assertFalse(has_more)

if __name__ == '__main__':
    unittest.main() 