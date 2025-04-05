from typing import Dict, List, Optional, Union, Any
import json
import time
from datetime import datetime
import logging

from src.utils import dict_to_json, generate_id
from src.graph_memory.base_manager import BaseManager
from src.graph_memory.embedding_adapter import EmbeddingAdapter

class LessonConsolidation:
    """
    Handles lesson memory consolidation processes.
    Consolidates related lessons, identifies patterns, and optimizes knowledge representation.
    """
    
    def __init__(self, base_manager: BaseManager):
        """
        Initialize the lesson consolidation manager.
        
        Args:
            base_manager: Base manager for graph operations
        """
        self.base_manager = base_manager
        self.logger = logging.getLogger(__name__)
        self.embedding_adapter = EmbeddingAdapter(logger=base_manager.logger)
    
    def identify_similar_lessons(self, min_similarity: float = 0.7, 
                              entity_type: Optional[str] = None,
                              max_results: int = 20) -> str:
        """
        Identify clusters of similar lessons based on semantic similarity.
        
        Args:
            min_similarity: Minimum similarity threshold (0.0-1.0)
            entity_type: Optional specific lesson type to filter by
            max_results: Maximum number of similarity pairs to return
            
        Returns:
            JSON string with similar lesson pairs
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Validate parameters
            if min_similarity < 0.0 or min_similarity > 1.0:
                min_similarity = 0.7
            
            if max_results < 1:
                max_results = 20
            elif max_results > 100:
                max_results = 100
            
            # Build query to get lessons with embeddings
            query_parts = ["MATCH (e:Entity)"]
            params = {}
            
            # Add domain filter
            query_parts.append("WHERE e.domain = 'lesson'")
            
            # Add type filter if specified
            if entity_type:
                query_parts.append("AND e.entityType = $entity_type")
                params["entity_type"] = entity_type
            
            # Ensure the entity has an embedding
            query_parts.append("AND e.embedding IS NOT NULL")
            
            # Return relevant fields
            query_parts.append("""
            RETURN e.id as id, 
                   e.name as name, 
                   e.entityType as type,
                   e.embedding as embedding,
                   e.confidence as confidence,
                   e.created as created
            """)
            
            # Execute query
            query = "\n".join(query_parts)
            # Use safe_execute_read_query for validation (read-only operation)
            records = self.base_manager.safe_execute_read_query(query, params)
            
            if not records:
                return dict_to_json({
                    "message": "No lessons with embeddings found",
                    "similar_pairs": []
                })
            
            # Process results
            lessons = []
            for record in records:
                lesson_id = record.get("id")
                name = record.get("name")
                lesson_type = record.get("type")
                embedding = record.get("embedding")
                confidence = record.get("confidence")
                created = record.get("created")
                
                if lesson_id and name and embedding:
                    lessons.append({
                        "id": lesson_id,
                        "name": name,
                        "type": lesson_type,
                        "embedding": embedding,
                        "confidence": confidence,
                        "created": created
                    })
            
            # Calculate similarities between lessons
            similar_pairs = []
            
            for i in range(len(lessons)):
                for j in range(i + 1, len(lessons)):
                    lesson1 = lessons[i]
                    lesson2 = lessons[j]
                    
                    # Skip comparing a lesson to itself
                    if lesson1["id"] == lesson2["id"]:
                        continue
                    
                    # Calculate similarity
                    similarity = self.embedding_adapter.similarity_score(
                        lesson1["embedding"], lesson2["embedding"]
                    )
                    
                    # Add to results if above threshold
                    if similarity >= min_similarity:
                        similar_pairs.append({
                            "lesson1": {
                                "id": lesson1["id"],
                                "name": lesson1["name"],
                                "type": lesson1["type"],
                                "confidence": lesson1["confidence"]
                            },
                            "lesson2": {
                                "id": lesson2["id"],
                                "name": lesson2["name"],
                                "type": lesson2["type"],
                                "confidence": lesson2["confidence"]
                            },
                            "similarity": similarity
                        })
            
            # Sort by similarity (highest first)
            similar_pairs.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Limit results
            if len(similar_pairs) > max_results:
                similar_pairs = similar_pairs[:max_results]
            
            # Return results
            return dict_to_json({
                "similar_pair_count": len(similar_pairs),
                "min_similarity_threshold": min_similarity,
                "similar_pairs": similar_pairs
            })
                
        except Exception as e:
            error_msg = f"Error identifying similar lessons: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def merge_lessons(self, source_lessons: List[Dict[str, Any]], 
                    new_name: str, 
                    merge_strategy: str = "latest",
                    container_name: Optional[str] = None) -> str:
        """
        Merge multiple related lessons into a consolidated lesson.
        
        Args:
            source_lessons: List of lesson entities to merge (must contain id or name)
            new_name: Name for the consolidated lesson
            merge_strategy: Strategy for merging (latest, weighted_average, or manual)
            container_name: Optional container to add the merged lesson to
            
        Returns:
            JSON string with the merged lesson details
        """
        try:
            self.base_manager.ensure_initialized()
            
            if not source_lessons or len(source_lessons) < 2:
                return dict_to_json({"error": "At least two source lessons are required for merging"})
            
            if not new_name:
                return dict_to_json({"error": "New name is required for the merged lesson"})
            
            # Validate merge strategy
            valid_strategies = ["latest", "weighted_average", "manual"]
            if merge_strategy not in valid_strategies:
                merge_strategy = "latest"
            
            # Extract source lesson identifiers
            source_ids = []
            source_names = []
            
            for lesson in source_lessons:
                if isinstance(lesson, dict):
                    if "id" in lesson:
                        source_ids.append(lesson["id"])
                    elif "name" in lesson:
                        source_names.append(lesson["name"])
                elif isinstance(lesson, str):
                    # Handle case where the lesson is provided as a string ID or name
                    if lesson.startswith("les_") or lesson.startswith("ent_"):
                        source_ids.append(lesson)
                    else:
                        source_names.append(lesson)
            
            # Build query to get lesson details
            query_parts = ["MATCH (e:Entity)"]
            params = {}
            
            # Build WHERE clause for source lessons
            where_clauses = []
            
            if source_ids:
                where_clauses.append("e.id IN $source_ids")
                params["source_ids"] = source_ids
            
            if source_names:
                where_clauses.append("e.name IN $source_names")
                params["source_names"] = source_names
            
            # Add domain filter
            where_clauses.append("e.domain = 'lesson'")
            
            query_parts.append("WHERE " + " OR ".join(where_clauses))
            
            # Return all relevant fields
            query_parts.append("""
            RETURN e.id as id,
                   e.name as name,
                   e.entityType as type,
                   e.content as content,
                   e.context as context,
                   e.created as created,
                   e.confidence as confidence,
                   e.embedding as embedding,
                   e.status as status,
                   e.properties as properties
            """)
            
            # Execute query
            query = "\n".join(query_parts)
            # Use safe_execute_read_query for validation (read-only operation)
            records = self.base_manager.safe_execute_read_query(query, params)
            
            if not records:
                return dict_to_json({"error": "No matching source lessons found"})
            
            # Process source lessons
            source_lesson_details = []
            for record in records:
                lesson_id = record.get("id")
                if lesson_id:
                    lesson_data = {k: v for k, v in record.items() if v is not None}
                    source_lesson_details.append(lesson_data)
            
            if len(source_lesson_details) < 2:
                return dict_to_json({"error": "At least two matching source lessons are required for merging"})
            
            # Apply merge strategy to create consolidated lesson
            merged_lesson = self._merge_lessons_by_strategy(
                source_lesson_details, new_name, merge_strategy
            )
            
            # Generate a new unique ID for the merged lesson
            merged_lesson["id"] = generate_id("les")
            merged_lesson["created"] = time.time()
            merged_lesson["domain"] = "lesson"
            merged_lesson["is_merged"] = True
            merged_lesson["source_count"] = len(source_lesson_details)
            
            # Add to database with relationships to source lessons
            create_query = """
            CREATE (m:Entity $merged_properties)
            RETURN m
            """
            
            # Use safe_execute_write_query for validation (write operation)
            create_records = self.base_manager.safe_execute_write_query(
                create_query, 
                {"merged_properties": merged_lesson}
            )
            
            if not create_records:
                return dict_to_json({"error": "Failed to create merged lesson"})
            
            # Create SUPERSEDES relationships from merged to source
            for source in source_lesson_details:
                source_id = source.get("id")
                if source_id:
                    supersedes_query = """
                    MATCH (m:Entity {id: $merged_id})
                    MATCH (s:Entity {id: $source_id})
                    CREATE (m)-[r:SUPERSEDES {created: $timestamp}]->(s)
                    """
                    
                    # Use safe_execute_write_query for validation (write operation)
                    self.base_manager.safe_execute_write_query(
                        supersedes_query,
                        {
                            "merged_id": merged_lesson["id"],
                            "source_id": source_id,
                            "timestamp": time.time()
                        }
                    )
            
            # Add to container if specified
            if container_name:
                # First check if container exists
                container_check_query = """
                MATCH (c:LessonContainer {name: $container_name})
                RETURN c
                """
                
                # Use safe_execute_read_query for validation (read-only operation)
                container_records = self.base_manager.safe_execute_read_query(
                    container_check_query,
                    {"container_name": container_name}
                )
                
                if container_records:
                    # Add to container
                    container_add_query = """
                    MATCH (c:LessonContainer {name: $container_name})
                    MATCH (m:Entity {id: $lesson_id})
                    CREATE (c)-[r:CONTAINS {added: $timestamp}]->(m)
                    """
                    
                    # Use safe_execute_write_query for validation (write operation)
                    self.base_manager.safe_execute_write_query(
                        container_add_query,
                        {
                            "container_name": container_name,
                            "lesson_id": merged_lesson["id"],
                            "timestamp": time.time()
                        }
                    )
            
            # Return the merged lesson with source info
            result = {
                "merged_lesson": merged_lesson,
                "source_lessons": [
                    {"id": s.get("id"), "name": s.get("name")} 
                    for s in source_lesson_details
                ],
                "merge_strategy": merge_strategy
            }
            
            return dict_to_json(result)
                
        except Exception as e:
            error_msg = f"Error merging lessons: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def suggest_consolidations(self, threshold: float = 0.8, max_suggestions: int = 10) -> str:
        """
        Suggest lessons that could be consolidated based on similarity.
        
        Args:
            threshold: Similarity threshold for suggestions (0.0-1.0)
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            JSON string with consolidation suggestions
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Validate parameters
            if threshold < 0.0 or threshold > 1.0:
                threshold = 0.8
            
            if max_suggestions < 1:
                max_suggestions = 10
            elif max_suggestions > 50:
                max_suggestions = 50
            
            # First, get similar lesson pairs
            similar_pairs_json = self.identify_similar_lessons(
                min_similarity=threshold,
                max_results=100
            )
            
            similar_pairs_data = json.loads(similar_pairs_json)
            similar_pairs = similar_pairs_data.get("similar_pairs", [])
            
            if not similar_pairs:
                return dict_to_json({
                    "message": "No similar lessons found that meet the threshold",
                    "suggestions": []
                })
            
            # Group by clusters using a simple algorithm
            # A cluster is a group of lessons that are all similar to each other
            clusters = []
            visited = set()
            
            for pair in similar_pairs:
                lesson1 = pair["lesson1"]["id"]
                lesson2 = pair["lesson2"]["id"]
                similarity = pair["similarity"]
                
                # Skip if both lessons have been processed
                if lesson1 in visited and lesson2 in visited:
                    continue
                
                # Check if either lesson is in an existing cluster
                found_cluster = False
                for cluster in clusters:
                    if lesson1 in cluster["lesson_ids"] or lesson2 in cluster["lesson_ids"]:
                        # Add both lessons to this cluster
                        cluster["lesson_ids"].add(lesson1)
                        cluster["lesson_ids"].add(lesson2)
                        cluster["lessons"].append(pair["lesson1"])
                        cluster["lessons"].append(pair["lesson2"])
                        cluster["similarity_sum"] += similarity
                        cluster["pair_count"] += 1
                        found_cluster = True
                        break
                
                # Create a new cluster if not found
                if not found_cluster:
                    clusters.append({
                        "lesson_ids": {lesson1, lesson2},
                        "lessons": [pair["lesson1"], pair["lesson2"]],
                        "similarity_sum": similarity,
                        "pair_count": 1
                    })
                
                # Mark as visited
                visited.add(lesson1)
                visited.add(lesson2)
            
            # Process clusters to remove duplicates and calculate average similarity
            unique_clusters = []
            for cluster in clusters:
                # Remove duplicate lessons by ID
                unique_lessons = {}
                for lesson in cluster["lessons"]:
                    if lesson["id"] not in unique_lessons:
                        unique_lessons[lesson["id"]] = lesson
                
                # Only consider clusters with multiple lessons
                if len(unique_lessons) >= 2:
                    avg_similarity = cluster["similarity_sum"] / cluster["pair_count"]
                    
                    unique_clusters.append({
                        "lessons": list(unique_lessons.values()),
                        "avg_similarity": avg_similarity,
                        "lesson_count": len(unique_lessons)
                    })
            
            # Sort clusters by average similarity
            unique_clusters.sort(key=lambda c: c["avg_similarity"], reverse=True)
            
            # Limit results
            if len(unique_clusters) > max_suggestions:
                unique_clusters = unique_clusters[:max_suggestions]
            
            # Generate suggested merged names
            for cluster in unique_clusters:
                lesson_names = [l["name"] for l in cluster["lessons"]]
                cluster["suggested_name"] = self._generate_merged_name(lesson_names)
            
            # Return results
            return dict_to_json({
                "suggestion_count": len(unique_clusters),
                "similarity_threshold": threshold,
                "suggestions": unique_clusters
            })
                
        except Exception as e:
            error_msg = f"Error suggesting consolidations: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def cleanup_superseded_lessons(self, older_than_days: int = 30, 
                                 min_confidence: float = 0.0,
                                 dry_run: bool = True) -> str:
        """
        Clean up lessons that have been superseded and are older than a given threshold.
        
        Args:
            older_than_days: Only include lessons older than this many days
            min_confidence: Only include lessons with confidence >= this value
            dry_run: If True, only report what would be done without making changes
            
        Returns:
            JSON string with cleanup details
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Validate parameters
            if older_than_days < 0:
                older_than_days = 30
            
            if min_confidence < 0.0 or min_confidence > 1.0:
                min_confidence = 0.0
            
            # Calculate cutoff timestamp
            now = time.time()
            cutoff_timestamp = now - (older_than_days * 24 * 60 * 60)
            
            # Query to find superseded lessons
            query = """
            MATCH (newer:Entity)-[:SUPERSEDES]->(older:Entity)
            WHERE older.created < $cutoff_timestamp
            AND older.domain = 'lesson'
            AND older.confidence >= $min_confidence
            RETURN older.id as id, older.name as name, 
                   older.created as created,
                   older.confidence as confidence,
                   newer.id as newer_id,
                   newer.name as newer_name
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            records = self.base_manager.safe_execute_read_query(
                query,
                {
                    "cutoff_timestamp": cutoff_timestamp,
                    "min_confidence": min_confidence
                }
            )
            
            if not records:
                return dict_to_json({
                    "message": "No superseded lessons found matching criteria",
                    "count": 0,
                    "dry_run": dry_run
                })
            
            # Process candidates for cleanup
            candidates = []
            for record in records:
                lesson_id = record.get("id")
                if lesson_id:
                    candidates.append({
                        "id": lesson_id,
                        "name": record.get("name"),
                        "created": record.get("created"),
                        "confidence": record.get("confidence"),
                        "newer_id": record.get("newer_id"),
                        "newer_name": record.get("newer_name")
                    })
            
            # If dry run, just return the candidates
            if dry_run:
                return dict_to_json({
                    "message": f"Found {len(candidates)} superseded lessons matching criteria",
                    "candidates": candidates,
                    "count": len(candidates),
                    "dry_run": True
                })
            
            # Process deletion
            deleted_count = 0
            failed_deletions = []
            
            for candidate in candidates:
                lesson_id = candidate["id"]
                
                # Delete the lesson
                delete_query = """
                MATCH (e:Entity {id: $lesson_id})
                DETACH DELETE e
                """
                
                try:
                    # Use safe_execute_write_query for validation (write operation)
                    self.base_manager.safe_execute_write_query(
                        delete_query,
                        {"lesson_id": lesson_id}
                    )
                    deleted_count += 1
                except Exception as e:
                    failed_deletions.append({
                        "id": lesson_id,
                        "name": candidate["name"],
                        "error": str(e)
                    })
            
            # Return results
            return dict_to_json({
                "message": f"Deleted {deleted_count} superseded lessons",
                "deleted_count": deleted_count,
                "failed_count": len(failed_deletions),
                "failed_deletions": failed_deletions,
                "dry_run": False
            })
                
        except Exception as e:
            error_msg = f"Error cleaning up superseded lessons: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def _merge_lessons_by_strategy(self, lessons: List[Dict[str, Any]], 
                                 new_name: str,
                                 strategy: str) -> Dict[str, Any]:
        """
        Apply the specified merge strategy to lessons.
        
        Args:
            lessons: List of lesson data to merge
            new_name: Name for the merged lesson
            strategy: Strategy to use for merging
            
        Returns:
            Dict with merged lesson data
        """
        if strategy == "weighted_average":
            return self._merge_weighted_average_strategy(lessons, new_name)
        else:
            # Default to latest
            return self._merge_latest_strategy(lessons, new_name)
            
    def _merge_latest_strategy(self, lessons: List[Dict[str, Any]], new_name: str) -> Dict[str, Any]:
        """
        Merge lessons using the 'latest' strategy - use the most recent lesson as base.
        
        Args:
            lessons: List of lesson data to merge
            new_name: Name for the merged lesson
            
        Returns:
            Dict with merged lesson data
        """
        # Sort lessons by created time, newest first
        sorted_lessons = sorted(lessons, key=lambda x: x.get("created", 0), reverse=True)
        latest_lesson = sorted_lessons[0]
        
        # Start with latest lesson as base
        merged = {
            "name": new_name,
            "entityType": latest_lesson.get("type", "concept"),
            "content": latest_lesson.get("content", ""),
            "context": latest_lesson.get("context", ""),
            "confidence": latest_lesson.get("confidence", 0.7),
            "status": latest_lesson.get("status", "active"),
            "properties": latest_lesson.get("properties", {})
        }
        
        # Include embedding if available
        if "embedding" in latest_lesson:
            merged["embedding"] = latest_lesson["embedding"]
        
        # Add source info
        sources = []
        for lesson in sorted_lessons:
            sources.append({
                "id": lesson.get("id"),
                "name": lesson.get("name"),
                "created": lesson.get("created")
            })
        
        merged["sources"] = sources
        
        # Add metadata about merge process
        merged["merge_type"] = "latest"
        merged["base_source"] = latest_lesson.get("id")
        
        return merged
    
    def _merge_weighted_average_strategy(self, lessons: List[Dict[str, Any]], new_name: str) -> Dict[str, Any]:
        """
        Merge lessons using the 'weighted_average' strategy. 
        Combines content based on confidence weighting.
        
        Args:
            lessons: List of lesson data to merge
            new_name: Name for the merged lesson
            
        Returns:
            Dict with merged lesson data
        """
        if not lessons:
            return {}
        
        # Calculate confidence sum for weighting
        confidence_sum = sum(lesson.get("confidence", 0.5) for lesson in lessons)
        
        # Default if no confidence values
        if confidence_sum <= 0:
            confidence_sum = len(lessons) * 0.5
            for lesson in lessons:
                lesson["confidence"] = 0.5
        
        # Sort lessons by created time, newest first
        sorted_lessons = sorted(lessons, key=lambda x: x.get("created", 0), reverse=True)
        latest_lesson = sorted_lessons[0]
        
        # Start with basic fields from the latest lesson
        merged = {
            "name": new_name,
            "entityType": latest_lesson.get("type", "concept"),
            "context": "",
            "content": "",
            "confidence": 0.0,
            "status": "active",
            "properties": {}
        }
        
        # Combine contexts
        contexts = set()
        for lesson in lessons:
            context = lesson.get("context", "")
            if context:
                contexts.add(context)
        
        merged["context"] = ", ".join(contexts)
        
        # Combine content and calculate confidence based on weighted average
        combined_content = []
        combined_confidence = 0.0
        
        for lesson in lessons:
            content = lesson.get("content", "")
            confidence = lesson.get("confidence", 0.5)
            
            if content:
                combined_content.append(content)
            
            # Add to weighted confidence
            weight = confidence / confidence_sum
            combined_confidence += confidence * weight
        
        merged["content"] = "\n\n".join(combined_content)
        merged["confidence"] = combined_confidence
        
        # Combine properties
        all_properties = {}
        for lesson in lessons:
            properties = lesson.get("properties", {})
            if isinstance(properties, dict):
                for key, value in properties.items():
                    if key not in all_properties:
                        all_properties[key] = value
        
        merged["properties"] = all_properties
        
        # Calculate average embedding if available
        embeddings = []
        for lesson in lessons:
            embedding = lesson.get("embedding")
            if embedding:
                confidence = lesson.get("confidence", 0.5)
                embeddings.append((embedding, confidence))
        
        if embeddings:
            # Weighted average of embeddings
            merged_embedding = None
            for embedding, confidence in embeddings:
                weight = confidence / confidence_sum
                
                if merged_embedding is None:
                    # Initialize with first embedding * weight
                    merged_embedding = [val * weight for val in embedding]
                else:
                    # Add weighted embedding values
                    for i in range(len(merged_embedding)):
                        merged_embedding[i] += embedding[i] * weight
            
            merged["embedding"] = merged_embedding
        
        # Add source info
        sources = []
        for lesson in sorted_lessons:
            sources.append({
                "id": lesson.get("id"),
                "name": lesson.get("name"),
                "created": lesson.get("created"),
                "confidence": lesson.get("confidence"),
                "weight": lesson.get("confidence", 0.5) / confidence_sum
            })
        
        merged["sources"] = sources
        
        # Add metadata about merge process
        merged["merge_type"] = "weighted_average"
        
        return merged
    
    def _generate_merged_name(self, names: List[str]) -> str:
        """
        Generate a suitable name for a merged lesson.
        
        Args:
            names: List of source lesson names
            
        Returns:
            Generated merged name
        """
        if not names:
            return "Merged Lesson"
        
        if len(names) == 1:
            return names[0]
        
        # If names are very similar, use the first one
        name_similarity = sum(1 for i in range(min(5, len(names[0]))) 
                              if i < len(names[1]) and names[0][i] == names[1][i])
        
        if name_similarity >= 3:
            return names[0]
        
        # Otherwise, combine the first two names
        if len(names) == 2:
            return f"Combined: {names[0]} & {names[1]}"
        else:
            return f"Merged: {names[0]} & {len(names)-1} more" 