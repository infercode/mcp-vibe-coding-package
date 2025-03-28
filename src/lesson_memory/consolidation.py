from typing import Dict, List, Optional, Union, Any
import json
import time
from datetime import datetime

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
        self.logger = base_manager.logger
        self.embedding_adapter = EmbeddingAdapter(base_manager)
    
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
            records, _ = self.base_manager.safe_execute_query(query, params)
            
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
                    similarity = self.embedding_adapter.calculate_similarity(
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
            records, _ = self.base_manager.safe_execute_query(query, params)
            
            if not records:
                return dict_to_json({"error": "No matching source lessons found"})
            
            # Process source lessons
            source_lesson_data = []
            for record in records:
                lesson_data = {
                    "id": record.get("id"),
                    "name": record.get("name"),
                    "type": record.get("type"),
                    "content": record.get("content"),
                    "context": record.get("context"),
                    "created": record.get("created"),
                    "confidence": record.get("confidence"),
                    "embedding": record.get("embedding"),
                    "status": record.get("status"),
                    "properties": record.get("properties", {})
                }
                source_lesson_data.append(lesson_data)
            
            # Check that we have enough lessons to merge
            if len(source_lesson_data) < 2:
                return dict_to_json({"error": f"Found only {len(source_lesson_data)} lessons, at least 2 required for merging"})
            
            # Get observations for source lessons
            for lesson in source_lesson_data:
                lesson_id = lesson["id"]
                obs_query = """
                MATCH (e:Entity {id: $id})-[:HAS_OBSERVATION]->(o:Observation)
                RETURN o.id as id,
                       o.type as type,
                       o.content as content,
                       o.created as created,
                       o.lastUpdated as lastUpdated
                """
                obs_records, _ = self.base_manager.safe_execute_query(
                    obs_query,
                    {"id": lesson_id}
                )
                
                observations = []
                if obs_records:
                    for record in obs_records:
                        obs = {
                            "id": record.get("id"),
                            "type": record.get("type"),
                            "content": record.get("content"),
                            "created": record.get("created"),
                            "lastUpdated": record.get("lastUpdated")
                        }
                        observations.append(obs)
                
                lesson["observations"] = observations
            
            # Merge lessons based on strategy
            merged_lesson = self._merge_lessons_by_strategy(
                source_lesson_data, 
                new_name,
                merge_strategy
            )
            
            # Create the merged lesson
            timestamp = time.time()
            merged_id = generate_id(prefix="les")
            
            create_query = """
            CREATE (e:Entity {
                id: $id,
                name: $name,
                entityType: $entityType,
                domain: 'lesson',
                content: $content,
                context: $context,
                created: $created,
                confidence: $confidence,
                status: $status,
                properties: $properties,
                merged: true,
                mergedFrom: $mergedFrom,
                mergeStrategy: $mergeStrategy
            })
            RETURN e.id as id
            """
            
            merged_source_ids = [l["id"] for l in source_lesson_data]
            
            create_params = {
                "id": merged_id,
                "name": new_name,
                "entityType": merged_lesson["type"],
                "content": merged_lesson["content"],
                "context": merged_lesson["context"],
                "created": timestamp,
                "confidence": merged_lesson["confidence"],
                "status": "active",
                "properties": merged_lesson["properties"],
                "mergedFrom": merged_source_ids,
                "mergeStrategy": merge_strategy
            }
            
            # Execute creation query
            result_records, _ = self.base_manager.safe_execute_query(
                create_query,
                create_params
            )
            
            if not result_records:
                return dict_to_json({"error": "Failed to create merged lesson"})
            
            # Generate embedding for merged lesson
            embedding_text = f"{new_name} {merged_lesson['content']} {merged_lesson['context']}"
            embedding = self.embedding_adapter.generate_embedding(embedding_text)
            
            if embedding:
                update_embedding_query = """
                MATCH (e:Entity {id: $id})
                SET e.embedding = $embedding
                """
                self.base_manager.safe_execute_query(
                    update_embedding_query,
                    {"id": merged_id, "embedding": embedding}
                )
            
            # Create observations for merged lesson
            for obs in merged_lesson["observations"]:
                obs_id = generate_id(prefix="obs")
                obs_query = """
                MATCH (e:Entity {id: $lesson_id})
                CREATE (o:Observation {
                    id: $id,
                    type: $type,
                    content: $content,
                    created: $created,
                    lastUpdated: $created
                })
                CREATE (e)-[:HAS_OBSERVATION]->(o)
                """
                obs_params = {
                    "lesson_id": merged_id,
                    "id": obs_id,
                    "type": obs["type"],
                    "content": obs["content"],
                    "created": timestamp
                }
                self.base_manager.safe_execute_query(obs_query, obs_params)
            
            # Create supersedes relationships to source lessons
            for source_id in merged_source_ids:
                supersedes_query = """
                MATCH (new:Entity {id: $new_id})
                MATCH (old:Entity {id: $old_id})
                CREATE (new)-[:SUPERSEDES {
                    created: $timestamp,
                    reason: 'merge'
                }]->(old)
                """
                supersedes_params = {
                    "new_id": merged_id,
                    "old_id": source_id,
                    "timestamp": timestamp
                }
                self.base_manager.safe_execute_query(supersedes_query, supersedes_params)
            
            # Add to container if specified
            if container_name:
                container_query = """
                MATCH (c:Entity {name: $container_name, entityType: 'LessonContainer'})
                MATCH (e:Entity {id: $lesson_id})
                MERGE (c)-[:CONTAINS {
                    created: $timestamp
                }]->(e)
                """
                container_params = {
                    "container_name": container_name,
                    "lesson_id": merged_id,
                    "timestamp": timestamp
                }
                self.base_manager.safe_execute_query(container_query, container_params)
            
            # Return the merged lesson
            return dict_to_json({
                "merged_lesson_id": merged_id,
                "merged_lesson_name": new_name,
                "source_lessons": len(source_lesson_data),
                "merge_strategy": merge_strategy,
                "observation_count": len(merged_lesson["observations"]),
                "message": f"Successfully merged {len(source_lesson_data)} lessons"
            })
                
        except Exception as e:
            error_msg = f"Error merging lessons: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def suggest_consolidations(self, threshold: float = 0.8, max_suggestions: int = 10) -> str:
        """
        Suggest groups of lessons that could be consolidated based on similarity.
        
        Args:
            threshold: Similarity threshold for suggesting consolidation
            max_suggestions: Maximum number of consolidation groups to suggest
            
        Returns:
            JSON string with consolidation suggestions
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Validate parameters
            if threshold < 0.5 or threshold > 1.0:
                threshold = 0.8
            
            if max_suggestions < 1:
                max_suggestions = 10
            elif max_suggestions > 50:
                max_suggestions = 50
            
            # First get similar lessons
            similar_lessons_json = self.identify_similar_lessons(
                min_similarity=threshold,
                max_results=100  # Get a larger set to find clusters from
            )
            
            try:
                similar_data = json.loads(similar_lessons_json)
                if "error" in similar_data:
                    return dict_to_json({"error": similar_data["error"]})
                
                similar_pairs = similar_data.get("similar_pairs", [])
                if not similar_pairs:
                    return dict_to_json({
                        "message": "No similar lessons found above threshold",
                        "suggestions": []
                    })
            except Exception as e:
                return dict_to_json({"error": f"Error processing similarity data: {str(e)}"})
            
            # Build graph from similar pairs
            lesson_connections = {}
            for pair in similar_pairs:
                lesson1 = pair["lesson1"]
                lesson2 = pair["lesson2"]
                similarity = pair["similarity"]
                
                l1_id = lesson1["id"]
                l2_id = lesson2["id"]
                
                # Add lesson1 to graph
                if l1_id not in lesson_connections:
                    lesson_connections[l1_id] = {
                        "lesson": lesson1,
                        "connected_to": {}
                    }
                
                # Add lesson2 to graph
                if l2_id not in lesson_connections:
                    lesson_connections[l2_id] = {
                        "lesson": lesson2,
                        "connected_to": {}
                    }
                
                # Add connections
                lesson_connections[l1_id]["connected_to"][l2_id] = similarity
                lesson_connections[l2_id]["connected_to"][l1_id] = similarity
            
            # Find clusters using a simple clustering algorithm
            visited = set()
            clusters = []
            
            for lesson_id in lesson_connections:
                if lesson_id in visited:
                    continue
                
                # Start a new cluster with this lesson
                cluster = []
                queue = [lesson_id]
                visited.add(lesson_id)
                
                while queue:
                    current_id = queue.pop(0)
                    cluster.append(current_id)
                    
                    # Find neighbors above threshold
                    for neighbor_id, similarity in lesson_connections[current_id]["connected_to"].items():
                        if neighbor_id not in visited and similarity >= threshold:
                            queue.append(neighbor_id)
                            visited.add(neighbor_id)
                
                # Only include clusters with at least 2 lessons
                if len(cluster) >= 2:
                    clusters.append(cluster)
            
            # Build suggestions
            suggestions = []
            for i, cluster in enumerate(clusters):
                if i >= max_suggestions:
                    break
                
                cluster_lessons = []
                lesson_names = []
                
                for lesson_id in cluster:
                    lesson_data = lesson_connections[lesson_id]["lesson"]
                    lesson_name = lesson_data["name"]
                    
                    cluster_lessons.append(lesson_data)
                    lesson_names.append(lesson_name)
                
                # Generate a suggested name based on common terms
                suggested_name = self._generate_merged_name(lesson_names)
                
                suggestion = {
                    "cluster_id": i + 1,
                    "lesson_count": len(cluster_lessons),
                    "lessons": cluster_lessons,
                    "suggested_name": suggested_name,
                    "reason": f"Group of {len(cluster_lessons)} lessons with high similarity (>= {threshold})"
                }
                suggestions.append(suggestion)
            
            return dict_to_json({
                "suggestion_count": len(suggestions),
                "similarity_threshold": threshold,
                "suggestions": suggestions
            })
                
        except Exception as e:
            error_msg = f"Error suggesting consolidations: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def cleanup_superseded_lessons(self, older_than_days: int = 30, 
                                 min_confidence: float = 0.0,
                                 dry_run: bool = True) -> str:
        """
        Archive superseded lessons that are no longer needed.
        
        Args:
            older_than_days: Only process lessons older than this many days
            min_confidence: Only process lessons with confidence below this threshold
            dry_run: If True, only report what would be changed without making changes
            
        Returns:
            JSON string with cleanup results
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Validate parameters
            if older_than_days < 0:
                older_than_days = 30
            
            if min_confidence < 0.0 or min_confidence > 1.0:
                min_confidence = 0.0
            
            # Calculate cutoff timestamp
            current_time = time.time()
            cutoff_timestamp = current_time - (older_than_days * 24 * 60 * 60)
            
            # Find superseded lessons
            query = """
            MATCH (newer:Entity)-[:SUPERSEDES]->(e:Entity)
            WHERE e.domain = 'lesson'
              AND e.created < $cutoff_timestamp
              AND e.confidence <= $min_confidence
            RETURN e.id as id,
                   e.name as name,
                   e.entityType as type,
                   e.created as created,
                   e.confidence as confidence,
                   newer.id as newer_id,
                   newer.name as newer_name
            """
            
            records, _ = self.base_manager.safe_execute_query(
                query,
                {
                    "cutoff_timestamp": cutoff_timestamp,
                    "min_confidence": min_confidence
                }
            )
            
            if not records:
                return dict_to_json({
                    "message": "No superseded lessons found matching criteria",
                    "candidates": []
                })
            
            # Process candidates
            candidates = []
            for record in records:
                lesson_id = record.get("id")
                name = record.get("name")
                lesson_type = record.get("type")
                created = record.get("created")
                confidence = record.get("confidence")
                newer_id = record.get("newer_id")
                newer_name = record.get("newer_name")
                
                created_date = datetime.fromtimestamp(created).strftime('%Y-%m-%d')
                days_old = int((current_time - created) / (24 * 60 * 60))
                
                candidate = {
                    "id": lesson_id,
                    "name": name,
                    "type": lesson_type,
                    "created": created,
                    "created_date": created_date,
                    "days_old": days_old,
                    "confidence": confidence,
                    "superseded_by": {
                        "id": newer_id,
                        "name": newer_name
                    }
                }
                candidates.append(candidate)
            
            # If dry run, just return candidates
            if dry_run:
                return dict_to_json({
                    "dry_run": True,
                    "message": f"Found {len(candidates)} superseded lessons that could be archived",
                    "candidates": candidates
                })
            
            # Otherwise, archive the candidates
            archived_count = 0
            
            for candidate in candidates:
                lesson_id = candidate["id"]
                
                # Archive by setting status to "archived"
                archive_query = """
                MATCH (e:Entity {id: $id})
                SET e.status = 'archived',
                    e.archivedDate = $timestamp,
                    e.archiveReason = 'superseded'
                """
                
                self.base_manager.safe_execute_query(
                    archive_query,
                    {
                        "id": lesson_id,
                        "timestamp": current_time
                    }
                )
                
                archived_count += 1
            
            return dict_to_json({
                "dry_run": False,
                "message": f"Archived {archived_count} superseded lessons",
                "archived_count": archived_count,
                "candidates": candidates
            })
                
        except Exception as e:
            error_msg = f"Error during cleanup of superseded lessons: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def _merge_lessons_by_strategy(self, lessons: List[Dict[str, Any]], 
                                 new_name: str,
                                 strategy: str) -> Dict[str, Any]:
        """
        Merge lessons using the specified strategy.
        
        Args:
            lessons: List of lesson data to merge
            new_name: Name for the merged lesson
            strategy: Merge strategy (latest, weighted_average, manual)
            
        Returns:
            Merged lesson data
        """
        if strategy == "latest":
            return self._merge_latest_strategy(lessons, new_name)
        elif strategy == "weighted_average":
            return self._merge_weighted_average_strategy(lessons, new_name)
        else:
            # Default to latest if strategy not recognized
            return self._merge_latest_strategy(lessons, new_name)
    
    def _merge_latest_strategy(self, lessons: List[Dict[str, Any]], new_name: str) -> Dict[str, Any]:
        """
        Merge lessons by taking fields from the most recent lesson.
        
        Args:
            lessons: List of lesson data to merge
            new_name: Name for the merged lesson
            
        Returns:
            Merged lesson data
        """
        # Sort by created timestamp (newest first)
        sorted_lessons = sorted(lessons, key=lambda x: x.get("created", 0), reverse=True)
        
        # Use the most recent lesson as the base
        latest_lesson = sorted_lessons[0]
        
        # Create merged lesson
        merged = {
            "name": new_name,
            "type": latest_lesson.get("type", "Lesson"),
            "content": latest_lesson.get("content", ""),
            "context": latest_lesson.get("context", ""),
            "confidence": latest_lesson.get("confidence", 0.5),
            "properties": dict(latest_lesson.get("properties", {}))
        }
        
        # Add source information to properties
        merged["properties"]["source_lessons"] = [l.get("name", "") for l in lessons]
        merged["properties"]["merge_date"] = time.time()
        
        # Collect unique observations from all lessons
        all_observations = {}
        for lesson in lessons:
            for obs in lesson.get("observations", []):
                obs_type = obs.get("type")
                if not obs_type:
                    continue
                
                # Only keep the latest observation of each type
                if obs_type not in all_observations:
                    all_observations[obs_type] = obs
                else:
                    existing_obs = all_observations[obs_type]
                    existing_updated = existing_obs.get("lastUpdated") or existing_obs.get("created", 0)
                    new_updated = obs.get("lastUpdated") or obs.get("created", 0)
                    
                    if new_updated > existing_updated:
                        all_observations[obs_type] = obs
        
        merged["observations"] = list(all_observations.values())
        
        return merged
    
    def _merge_weighted_average_strategy(self, lessons: List[Dict[str, Any]], new_name: str) -> Dict[str, Any]:
        """
        Merge lessons using weighted average based on confidence and recency.
        
        Args:
            lessons: List of lesson data to merge
            new_name: Name for the merged lesson
            
        Returns:
            Merged lesson data
        """
        if not lessons:
            return {
                "name": new_name,
                "type": "Lesson",
                "content": "",
                "context": "",
                "confidence": 0.5,
                "properties": {},
                "observations": []
            }
        
        # Get current time for calculating age weights
        current_time = time.time()
        
        # Calculate weights for each lesson based on confidence and recency
        total_weight = 0
        lesson_weights = []
        
        for lesson in lessons:
            confidence = lesson.get("confidence", 0.5)
            created = lesson.get("created", current_time)
            
            # Calculate age in days
            age_days = max(1, (current_time - created) / (24 * 60 * 60))
            
            # Calculate recency factor (more recent = higher weight)
            recency_factor = 1.0 / max(1, age_days / 30)  # Normalize by month
            
            # Combined weight (confidence * recency)
            weight = confidence * recency_factor
            
            lesson_weights.append(weight)
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in lesson_weights]
        else:
            # Equal weights if total weight is 0
            normalized_weights = [1.0 / len(lessons) for _ in lessons]
        
        # Determine most common type
        type_counts = {}
        for lesson in lessons:
            lesson_type = lesson.get("type", "Lesson")
            type_counts[lesson_type] = type_counts.get(lesson_type, 0) + 1
        
        most_common_type = max(type_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate weighted confidence
        weighted_confidence = sum(
            lesson.get("confidence", 0.5) * weight 
            for lesson, weight in zip(lessons, normalized_weights)
        )
        
        # Combine contexts with weights
        contexts = [lesson.get("context", "") for lesson in lessons]
        contents = [lesson.get("content", "") for lesson in lessons]
        
        # For content and context, we'll use the most heavily weighted lesson
        max_weight_index = normalized_weights.index(max(normalized_weights))
        primary_context = contexts[max_weight_index]
        primary_content = contents[max_weight_index]
        
        # Merge properties with priority to higher-weighted lessons
        merged_properties = {}
        for lesson, weight in sorted(zip(lessons, normalized_weights), key=lambda x: x[1], reverse=True):
            for key, value in lesson.get("properties", {}).items():
                if key not in merged_properties:
                    merged_properties[key] = value
        
        # Add source information to properties
        merged_properties["source_lessons"] = [l.get("name", "") for l in lessons]
        merged_properties["merge_date"] = time.time()
        merged_properties["weighted_merge_confidence"] = weighted_confidence
        
        # Create merged lesson
        merged = {
            "name": new_name,
            "type": most_common_type,
            "content": primary_content,
            "context": primary_context,
            "confidence": weighted_confidence,
            "properties": merged_properties
        }
        
        # Collect unique observations from all lessons, with priority to higher-weighted lessons
        all_observations = {}
        for lesson, weight in sorted(zip(lessons, normalized_weights), key=lambda x: x[1], reverse=True):
            for obs in lesson.get("observations", []):
                obs_type = obs.get("type")
                if not obs_type:
                    continue
                
                # Keep observation if not already included or if from a higher-weighted lesson
                if obs_type not in all_observations:
                    all_observations[obs_type] = obs
        
        merged["observations"] = list(all_observations.values())
        
        return merged
    
    def _generate_merged_name(self, names: List[str]) -> str:
        """
        Generate a new name for a merged lesson based on common terms in source lessons.
        
        Args:
            names: List of source lesson names
            
        Returns:
            Suggested name for the merged lesson
        """
        if not names:
            return "Consolidated Lesson"
        
        if len(names) == 1:
            return names[0]
        
        # Tokenize names and find common words
        tokenized_names = []
        for name in names:
            # Simple tokenization by splitting on spaces and removing punctuation
            tokens = "".join(c if c.isalnum() or c.isspace() else " " for c in name.lower()).split()
            tokenized_names.append(set(tokens))
        
        # Find common tokens
        common_tokens = set.intersection(*tokenized_names)
        
        # Remove very common words
        stop_words = {"the", "a", "an", "and", "or", "but", "of", "in", "on", "for", "to", "with", "lesson"}
        common_tokens = common_tokens - stop_words
        
        if common_tokens:
            # Reconstruct name from common tokens
            common_name = " ".join(sorted(common_tokens)).title()
            return f"Consolidated {common_name} Lesson"
        
        # Fall back to a generic name
        return f"Consolidated Lesson ({len(names)} sources)" 