from typing import Any, Dict, List, Optional, Union
import time
import json
import logging
import uuid

from src.utils import dict_to_json, generate_id
from src.graph_memory.base_manager import BaseManager
from src.graph_memory.observation_manager import ObservationManager

class LessonObservation:
    """
    Manager for lesson observation operations.
    Extends the core observation manager with lesson-specific functionality.
    """
    
    # Define standard lesson observation types
    LESSON_OBSERVATION_TYPES = [
        "WhatWasLearned",     # Factual knowledge gained
        "WhyItMatters",       # Importance and consequences
        "HowToApply",         # Application guidance
        "RootCause",          # Underlying causes
        "Evidence"            # Examples and supporting data
    ]
    
    def __init__(self, base_manager: BaseManager):
        """
        Initialize the lesson observation manager.
        
        Args:
            base_manager: Base manager for graph operations
        """
        self.base_manager = base_manager
        self.logger = logging.getLogger(__name__)
        self.observation_manager = ObservationManager(base_manager)
    
    def add_lesson_observation(self, entity_name: str, content: str, 
                            observation_type: str = "observation",
                            confidence: float = 1.0,
                            properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Add an observation to a lesson entity.
        
        Args:
            entity_name: Name of the entity to add the observation to
            content: Text content of the observation
            observation_type: Type of observation
            confidence: Confidence score (0.0 to 1.0)
            properties: Optional additional properties
            
        Returns:
            JSON string with the created observation
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if entity exists
            entity_query = """
            MATCH (e:Entity {name: $entity_name})
            RETURN e
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            entity_records = self.base_manager.safe_execute_read_query(
                entity_query,
                {"entity_name": entity_name}
            )
            
            if not entity_records or len(entity_records) == 0:
                return dict_to_json({"error": f"Entity '{entity_name}' not found"})
            
            # Check if entity belongs to the lesson domain
            entity = dict(entity_records[0]["e"].items())
            if entity.get("domain") != "lesson":
                return dict_to_json({
                    "error": f"Entity '{entity_name}' is not a lesson entity"
                })
            
            # Generate observation ID
            observation_id = str(uuid.uuid4())
            timestamp = time.time()
            
            # Prepare observation properties
            observation_props = properties or {}
            observation_props.update({
                "id": observation_id,
                "content": content,
                "type": observation_type,
                "created": timestamp,
                "lastUpdated": timestamp,
                "domain": "lesson",  # Mark as lesson observation
                "confidence": max(0.0, min(1.0, confidence))  # Clamp to 0-1 range
            })
            
            # Create observation
            create_query = """
            CREATE (o:Observation $properties)
            RETURN o
            """
            
            # Use safe_execute_write_query for validation (write operation)
            create_records = self.base_manager.safe_execute_write_query(
                create_query,
                {"properties": observation_props}
            )
            
            if not create_records or len(create_records) == 0:
                return dict_to_json({"error": "Failed to create observation"})
            
            # Link observation to entity
            link_query = """
            MATCH (e:Entity {name: $entity_name})
            MATCH (o:Observation {id: $observation_id})
            CREATE (e)-[r:HAS_OBSERVATION {created: $timestamp}]->(o)
            RETURN o
            """
            
            # Use safe_execute_write_query for validation (write operation)
            link_records = self.base_manager.safe_execute_write_query(
                link_query,
                {
                    "entity_name": entity_name,
                    "observation_id": observation_id,
                    "timestamp": timestamp
                }
            )
            
            if not link_records or len(link_records) == 0:
                # Attempt to clean up the created observation
                self.base_manager.safe_execute_write_query(
                    "MATCH (o:Observation {id: $id}) DELETE o",
                    {"id": observation_id}
                )
                return dict_to_json({"error": "Failed to link observation to entity"})
            
            # Get the created observation
            observation = dict(link_records[0]["o"].items())
            
            self.logger.info(f"Added lesson observation to entity '{entity_name}'")
            return dict_to_json({
                "observation": observation,
                "message": f"Observation added to entity '{entity_name}'"
            })
                
        except Exception as e:
            error_msg = f"Error adding lesson observation: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def get_lesson_observations(self, entity_name: str, observation_type: Optional[str] = None) -> str:
        """
        Get observations for a lesson entity.
        
        Args:
            entity_name: Name of the entity
            observation_type: Optional observation type to filter by
            
        Returns:
            JSON string with the observations
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if entity exists
            entity_query = """
            MATCH (e:Entity {name: $entity_name})
            RETURN e
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            entity_records = self.base_manager.safe_execute_read_query(
                entity_query,
                {"entity_name": entity_name}
            )
            
            if not entity_records or len(entity_records) == 0:
                return dict_to_json({"error": f"Entity '{entity_name}' not found"})
            
            # Check if entity belongs to the lesson domain
            entity = dict(entity_records[0]["e"].items())
            if entity.get("domain") != "lesson":
                return dict_to_json({
                    "error": f"Entity '{entity_name}' is not a lesson entity"
                })
            
            # Build query based on observation_type filter
            if observation_type:
                query = """
                MATCH (e:Entity {name: $entity_name})-[:HAS_OBSERVATION]->(o:Observation)
                WHERE o.type = $observation_type
                RETURN o
                ORDER BY o.created DESC
                """
                params = {"entity_name": entity_name, "observation_type": observation_type}
            else:
                query = """
                MATCH (e:Entity {name: $entity_name})-[:HAS_OBSERVATION]->(o:Observation)
                RETURN o
                ORDER BY o.created DESC
                """
                params = {"entity_name": entity_name}
            
            # Use safe_execute_read_query for validation (read-only operation)
            records = self.base_manager.safe_execute_read_query(query, params)
            
            # Process results
            observations = []
            if records:
                for record in records:
                    observation = dict(record["o"].items())
                    observations.append(observation)
            
            return dict_to_json({
                "entity": entity_name,
                "observation_count": len(observations),
                "observation_type": observation_type,
                "observations": observations
            })
                
        except Exception as e:
            error_msg = f"Error retrieving lesson observations: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def update_lesson_observation(self, observation_id: str, updates: Dict[str, Any]) -> str:
        """
        Update a lesson observation.
        
        Args:
            observation_id: ID of the observation to update
            updates: Dictionary of property updates
            
        Returns:
            JSON string with the updated observation
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if observation exists and is a lesson observation
            observation_query = """
            MATCH (o:Observation {id: $observation_id})
            RETURN o
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            observation_records = self.base_manager.safe_execute_read_query(
                observation_query,
                {"observation_id": observation_id}
            )
            
            if not observation_records or len(observation_records) == 0:
                return dict_to_json({"error": f"Observation with ID '{observation_id}' not found"})
            
            # Check if it's a lesson observation
            observation = dict(observation_records[0]["o"].items())
            if observation.get("domain") != "lesson":
                return dict_to_json({
                    "error": f"Observation '{observation_id}' is not a lesson observation"
                })
            
            # Validate updates - prevent changing core properties
            protected_fields = ["id", "created", "domain"]
            for field in protected_fields:
                if field in updates:
                    del updates[field]
            
            if not updates:
                return dict_to_json({
                    "observation": observation,
                    "message": "No valid updates provided"
                })
            
            # Add lastUpdated timestamp
            updates["lastUpdated"] = time.time()
            
            # If confidence is being updated, ensure it's in the valid range
            if "confidence" in updates:
                updates["confidence"] = max(0.0, min(1.0, updates["confidence"]))
            
            # Prepare update parts
            set_parts = []
            for key, value in updates.items():
                set_parts.append(f"o.{key} = ${key}")
            
            # Build update query
            update_query = f"""
            MATCH (o:Observation {{id: $observation_id}})
            SET {', '.join(set_parts)}
            RETURN o
            """
            
            # Add observation_id to updates for the query
            params = {"observation_id": observation_id, **updates}
            
            # Use safe_execute_write_query for validation (write operation)
            update_records = self.base_manager.safe_execute_write_query(
                update_query,
                params
            )
            
            if not update_records or len(update_records) == 0:
                return dict_to_json({"error": f"Failed to update observation '{observation_id}'"})
            
            # Return updated observation
            updated_observation = dict(update_records[0]["o"].items())
            
            self.logger.info(f"Updated lesson observation: {observation_id}")
            return dict_to_json({
                "observation": updated_observation,
                "message": "Observation updated successfully"
            })
                
        except Exception as e:
            error_msg = f"Error updating lesson observation: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def delete_lesson_observation(self, observation_id: str) -> str:
        """
        Delete a lesson observation.
        
        Args:
            observation_id: ID of the observation to delete
            
        Returns:
            JSON string with the deletion result
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if observation exists and is a lesson observation
            observation_query = """
            MATCH (o:Observation {id: $observation_id})
            RETURN o
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            observation_records = self.base_manager.safe_execute_read_query(
                observation_query,
                {"observation_id": observation_id}
            )
            
            if not observation_records or len(observation_records) == 0:
                return dict_to_json({"error": f"Observation with ID '{observation_id}' not found"})
            
            # Check if it's a lesson observation
            observation = dict(observation_records[0]["o"].items())
            if observation.get("domain") != "lesson":
                return dict_to_json({
                    "error": f"Observation '{observation_id}' is not a lesson observation"
                })
            
            # Delete relationships first
            delete_rels_query = """
            MATCH (o:Observation {id: $observation_id})
            MATCH (e)-[r:HAS_OBSERVATION]->(o)
            DELETE r
            """
            
            # Use safe_execute_write_query for validation (write operation)
            self.base_manager.safe_execute_write_query(
                delete_rels_query,
                {"observation_id": observation_id}
            )
            
            # Delete the observation
            delete_query = """
            MATCH (o:Observation {id: $observation_id})
            DELETE o
            """
            
            # Use safe_execute_write_query for validation (write operation)
            self.base_manager.safe_execute_write_query(
                delete_query,
                {"observation_id": observation_id}
            )
            
            self.logger.info(f"Deleted lesson observation: {observation_id}")
            return dict_to_json({
                "status": "success",
                "message": f"Observation '{observation_id}' deleted successfully"
            })
                
        except Exception as e:
            error_msg = f"Error deleting lesson observation: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def create_structured_lesson_observations(self, entity_name: str,
                                           what_was_learned: Optional[str] = None,
                                           why_it_matters: Optional[str] = None,
                                           how_to_apply: Optional[str] = None,
                                           root_cause: Optional[str] = None,
                                           evidence: Optional[str] = None,
                                           container_name: Optional[str] = None) -> str:
        """
        Create all structured observations for a lesson entity at once.
        
        Args:
            entity_name: Name of the entity
            what_was_learned: Optional content for WhatWasLearned observation
            why_it_matters: Optional content for WhyItMatters observation
            how_to_apply: Optional content for HowToApply observation
            root_cause: Optional content for RootCause observation
            evidence: Optional content for Evidence observation
            container_name: Optional container to verify entity membership
            
        Returns:
            JSON string with the created observations
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if entity exists and is in the container
            if container_name:
                container_query = """
                MATCH (c:LessonContainer {name: $container_name})-[:CONTAINS]->(e:Entity {name: $entity_name})
                RETURN e
                """
                
                container_records = self.base_manager.safe_execute_read_query(
                    container_query,
                    {"container_name": container_name, "entity_name": entity_name}
                )
                
                if not container_records or len(container_records) == 0:
                    return dict_to_json({
                        "error": f"Entity '{entity_name}' not found in container '{container_name}'"
                    })
            
            # Build list of observations to create
            observations = []
            
            if what_was_learned:
                observations.append({
                    "entity": entity_name,
                    "content": what_was_learned,
                    "type": "WhatWasLearned"
                })
                
            if why_it_matters:
                observations.append({
                    "entity": entity_name,
                    "content": why_it_matters,
                    "type": "WhyItMatters"
                })
                
            if how_to_apply:
                observations.append({
                    "entity": entity_name,
                    "content": how_to_apply,
                    "type": "HowToApply"
                })
                
            if root_cause:
                observations.append({
                    "entity": entity_name,
                    "content": root_cause,
                    "type": "RootCause"
                })
                
            if evidence:
                observations.append({
                    "entity": entity_name,
                    "content": evidence,
                    "type": "Evidence"
                })
            
            if not observations:
                return dict_to_json({
                    "error": "No observation content provided"
                })
            
            # Add observations
            return self.observation_manager.add_observations(observations)
                
        except Exception as e:
            error_msg = f"Error creating structured lesson observations: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def check_observation_completeness(self, entity_name: str) -> str:
        """
        Check which structured observation types are present for a lesson entity.
        
        Args:
            entity_name: Name of the entity
            
        Returns:
            JSON string with completeness assessment
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Get all observation types for the entity
            query = """
            MATCH (e:Entity {name: $entity_name})-[:HAS_OBSERVATION]->(o:Observation)
            RETURN DISTINCT o.type as type
            """
            
            records = self.base_manager.safe_execute_read_query(
                query,
                {"entity_name": entity_name}
            )
            
            # Extract types
            existing_types = set()
            if records:
                for record in records:
                    obs_type = record.get("type")
                    if obs_type:
                        existing_types.add(obs_type)
            
            # Check completeness
            completeness = {}
            for obs_type in self.LESSON_OBSERVATION_TYPES:
                completeness[obs_type] = obs_type in existing_types
            
            # Calculate overall completeness score
            complete_count = sum(1 for present in completeness.values() if present)
            total_count = len(self.LESSON_OBSERVATION_TYPES)
            completeness_score = complete_count / total_count if total_count > 0 else 0.0
            
            return dict_to_json({
                "entity": entity_name,
                "completeness": completeness,
                "completeness_score": completeness_score,
                "missing_types": [t for t in self.LESSON_OBSERVATION_TYPES if t not in existing_types]
            })
                
        except Exception as e:
            error_msg = f"Error checking observation completeness: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg}) 