from typing import Any, Dict, List, Optional, Union
import time
import json

from src.utils import dict_to_json, generate_id
from src.graph_memory.base_manager import BaseManager
from src.graph_memory.observation_manager import ObservationManager

class LessonObservation:
    """
    Manager for structured lesson observations.
    Handles typed observations with specific semantic meaning.
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
        self.logger = base_manager.logger
        self.observation_manager = ObservationManager(base_manager)
    
    def add_lesson_observation(self, entity_name: str, content: str, observation_type: str,
                            container_name: Optional[str] = None) -> str:
        """
        Add a structured observation to a lesson entity.
        
        Args:
            entity_name: Name of the entity to add observation to
            content: Content of the observation
            observation_type: Type of observation (should be a lesson observation type)
            container_name: Optional container to verify entity membership
            
        Returns:
            JSON string with the added observation
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Validate observation type
            if observation_type not in self.LESSON_OBSERVATION_TYPES:
                self.logger.info(f"Observation type '{observation_type}' is not a standard lesson observation type")
            
            # Check if entity exists and is in the container
            if container_name:
                container_query = """
                MATCH (c:LessonContainer {name: $container_name})-[:CONTAINS]->(e:Entity {name: $entity_name})
                RETURN e
                """
                
                container_records, _ = self.base_manager.safe_execute_query(
                    container_query,
                    {"container_name": container_name, "entity_name": entity_name}
                )
                
                if not container_records or len(container_records) == 0:
                    return dict_to_json({
                        "error": f"Entity '{entity_name}' not found in container '{container_name}'"
                    })
            
            # Create the observation
            observation_dict = {
                "entity": entity_name,
                "content": content,
                "type": observation_type
            }
            
            # Add structured observations to entity
            observation_result = self.observation_manager.add_observations([observation_dict])
            return observation_result
                
        except Exception as e:
            error_msg = f"Error adding lesson observation: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def get_lesson_observations(self, entity_name: str, observation_type: Optional[str] = None,
                             container_name: Optional[str] = None) -> str:
        """
        Get structured observations for a lesson entity.
        
        Args:
            entity_name: Name of the entity
            observation_type: Optional type to filter by
            container_name: Optional container to verify entity membership
            
        Returns:
            JSON string with the observations
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if entity exists and is in the container
            if container_name:
                container_query = """
                MATCH (c:LessonContainer {name: $container_name})-[:CONTAINS]->(e:Entity {name: $entity_name})
                RETURN e
                """
                
                container_records, _ = self.base_manager.safe_execute_query(
                    container_query,
                    {"container_name": container_name, "entity_name": entity_name}
                )
                
                if not container_records or len(container_records) == 0:
                    return dict_to_json({
                        "error": f"Entity '{entity_name}' not found in container '{container_name}'"
                    })
            
            # Build query based on observation_type
            query_parts = ["MATCH (e:Entity {name: $entity_name})-[:HAS_OBSERVATION]->(o:Observation)"]
            params = {"entity_name": entity_name}
            
            if observation_type:
                query_parts.append("WHERE o.type = $observation_type")
                params["observation_type"] = observation_type
                
                # Allow filtering by lesson observation type group
                if observation_type in self.LESSON_OBSERVATION_TYPES:
                    query_parts[-1] = "WHERE o.type = $observation_type"
                
            # Complete query
            query_parts.append("RETURN o.id as id, o.content as content, o.type as type, o.created as created")
            query = "\n".join(query_parts)
            
            # Execute query
            records, _ = self.base_manager.safe_execute_query(
                query,
                params
            )
            
            # Process results
            observations = []
            if records:
                for record in records:
                    observation = {
                        "id": record.get("id"),
                        "content": record.get("content"),
                        "type": record.get("type"),
                        "created": record.get("created")
                    }
                    observations.append(observation)
            
            # Format results by type if no specific type is requested
            if observation_type is None:
                # Group observations by type
                observations_by_type = {}
                for obs in observations:
                    obs_type = obs.get("type")
                    if obs_type not in observations_by_type:
                        observations_by_type[obs_type] = []
                    observations_by_type[obs_type].append(obs)
                
                return dict_to_json({
                    "entity": entity_name,
                    "observations_by_type": observations_by_type
                })
            
            return dict_to_json({
                "entity": entity_name,
                "observations": observations
            })
                
        except Exception as e:
            error_msg = f"Error retrieving lesson observations: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def update_lesson_observation(self, entity_name: str, observation_id: str, 
                               content: str, observation_type: Optional[str] = None) -> str:
        """
        Update a lesson observation.
        
        Args:
            entity_name: Name of the entity
            observation_id: ID of the observation to update
            content: New content for the observation
            observation_type: Optional new type for the observation
            
        Returns:
            JSON string with the updated observation
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Validate observation type if provided
            if observation_type and observation_type not in self.LESSON_OBSERVATION_TYPES:
                self.logger.info(f"Observation type '{observation_type}' is not a standard lesson observation type")
            
            # Update the observation
            return self.observation_manager.update_observation(
                entity_name, observation_id, content, observation_type
            )
                
        except Exception as e:
            error_msg = f"Error updating lesson observation: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def delete_lesson_observation(self, entity_name: str, observation_id: str) -> str:
        """
        Delete a lesson observation.
        
        Args:
            entity_name: Name of the entity
            observation_id: ID of the observation to delete
            
        Returns:
            JSON string with the deletion result
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Delete the observation
            return self.observation_manager.delete_observation(
                entity_name, observation_id=observation_id
            )
                
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
                
                container_records, _ = self.base_manager.safe_execute_query(
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
            
            records, _ = self.base_manager.safe_execute_query(
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