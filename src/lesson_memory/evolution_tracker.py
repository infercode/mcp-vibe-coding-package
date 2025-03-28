from typing import Any, Dict, List, Optional, Union
import time
import json
from datetime import datetime, timezone

from src.utils import dict_to_json, generate_id
from src.graph_memory.base_manager import BaseManager

class EvolutionTracker:
    """
    Tracker for lesson knowledge evolution over time.
    Analyzes how knowledge has evolved with temporal intelligence.
    """
    
    def __init__(self, base_manager: BaseManager):
        """
        Initialize the evolution tracker.
        
        Args:
            base_manager: Base manager for graph operations
        """
        self.base_manager = base_manager
        self.logger = base_manager.logger
    
    def get_knowledge_evolution(self, entity_name: Optional[str] = None, 
                             lesson_type: Optional[str] = None,
                             start_date: Optional[Union[str, float]] = None,
                             end_date: Optional[Union[str, float]] = None,
                             include_superseded: bool = True) -> str:
        """
        Track how knowledge has evolved over time.
        
        Args:
            entity_name: Optional entity name to filter by
            lesson_type: Optional lesson type to filter by
            start_date: Optional start date (ISO string or timestamp)
            end_date: Optional end date (ISO string or timestamp)
            include_superseded: Whether to include superseded lessons
            
        Returns:
            JSON string with the knowledge evolution timeline
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Process date parameters
            date_params = {}
            
            if start_date:
                if isinstance(start_date, str):
                    # Convert ISO format to timestamp
                    try:
                        dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                        date_params["start_timestamp"] = dt.timestamp()
                    except (ValueError, TypeError):
                        self.logger.error(f"Invalid start_date format: {start_date}")
                        return dict_to_json({"error": "Invalid start_date format. Use ISO format (YYYY-MM-DD)."})
                else:
                    date_params["start_timestamp"] = start_date
            
            if end_date:
                if isinstance(end_date, str):
                    # Convert ISO format to timestamp
                    try:
                        dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                        date_params["end_timestamp"] = dt.timestamp()
                    except (ValueError, TypeError):
                        self.logger.error(f"Invalid end_date format: {end_date}")
                        return dict_to_json({"error": "Invalid end_date format. Use ISO format (YYYY-MM-DD)."})
                else:
                    date_params["end_timestamp"] = end_date
            
            # Build query parts and parameters
            query_parts = []
            params = {}
            
            # Base lesson entity matching
            query_parts.append("MATCH (l:Entity)")
            query_parts.append("WHERE l.domain = 'lesson'")
            
            # Name filter
            if entity_name:
                query_parts.append("AND (l.name CONTAINS $entity_name OR l.context CONTAINS $entity_name)")
                params["entity_name"] = entity_name
            
            # Type filter
            if lesson_type:
                query_parts.append("AND l.entityType = $lesson_type")
                params["lesson_type"] = lesson_type
            
            # Handle superseded lessons
            if not include_superseded:
                query_parts.append("AND NOT EXISTS { MATCH (l)<-[:SUPERSEDES]-() }")
            
            # Get version history paths
            query_parts.append("WITH l")
            query_parts.append("OPTIONAL MATCH path = (l)-[:SUPERSEDES*0..]->(old:Entity)")
            query_parts.append("WITH l, old, length(path) as distance, path")
            
            # Date filters
            if "start_timestamp" in date_params:
                query_parts.append("WHERE distance = 0 OR old.created >= $start_timestamp")
                params["start_timestamp"] = date_params["start_timestamp"]
            
            if "end_timestamp" in date_params:
                if "start_timestamp" in date_params:
                    query_parts.append("AND (distance = 0 OR old.created <= $end_timestamp)")
                else:
                    query_parts.append("WHERE distance = 0 OR old.created <= $end_timestamp")
                params["end_timestamp"] = date_params["end_timestamp"]
            
            # Collect lesson data and relationships in timeline
            query_parts.append("""
            WITH l, collect({
              id: old.id,
              name: old.name,
              version: old.version,
              created: old.created,
              confidence: old.confidence,
              status: old.status,
              relationships: [(old)-[r]->(related) WHERE r.domain = 'lesson' | {
                type: type(r),
                target: related.name,
                properties: properties(r)
              }]
            }) as versions
            """)
            
            # Get application events
            query_parts.append("""
            OPTIONAL MATCH (l)-[applied:APPLIED_TO]->(target)
            WITH l, versions, collect({
              target: target.name,
              date: applied.applied_date,
              success_score: applied.success_score,
              notes: applied.application_notes
            }) as applications
            """)
            
            # Get observation updates
            query_parts.append("""
            OPTIONAL MATCH (l)-[:HAS_OBSERVATION]->(o:Observation)
            WITH l, versions, applications, collect({
              type: o.type,
              content: o.content,
              created: o.created,
              lastUpdated: o.lastUpdated
            }) as observations
            """)
            
            # Format final results
            query_parts.append("""
            RETURN l.name as lesson_name,
                   l.id as id,
                   l.entityType as type,
                   l.created as created,
                   l.confidence as confidence,
                   l.context as context,
                   versions,
                   applications,
                   observations
            ORDER BY l.created DESC
            """)
            
            # Complete query
            query = "\n".join(query_parts)
            
            # Execute query
            records, _ = self.base_manager.safe_execute_query(
                query,
                params
            )
            
            # Process results
            results = []
            if records:
                for record in records:
                    lesson_data = {
                        "name": record.get("lesson_name"),
                        "id": record.get("id"),
                        "type": record.get("type"),
                        "created": record.get("created"),
                        "confidence": record.get("confidence"),
                        "context": record.get("context"),
                    }
                    
                    # Add version history
                    versions = record.get("versions", [])
                    if versions:
                        # Sort versions by creation date
                        sorted_versions = sorted(versions, key=lambda v: v.get("created", 0))
                        lesson_data["version_history"] = sorted_versions
                    
                    # Add application history
                    applications = record.get("applications", [])
                    if applications:
                        # Sort applications by date
                        sorted_applications = sorted(applications, key=lambda a: a.get("date", 0))
                        lesson_data["application_history"] = sorted_applications
                    
                    # Add observation history
                    observations = record.get("observations", [])
                    if observations:
                        # Sort observations by creation date
                        sorted_observations = sorted(observations, key=lambda o: o.get("created", 0))
                        lesson_data["observation_history"] = sorted_observations
                    
                    # Generate unified timeline
                    timeline = []
                    
                    # Add version events
                    for version in versions:
                        if version.get("created"):
                            timeline.append({
                                "timestamp": version.get("created"),
                                "type": "VERSION",
                                "version": version.get("version"),
                                "confidence": version.get("confidence"),
                                "status": version.get("status")
                            })
                    
                    # Add application events
                    for application in applications:
                        if application.get("date"):
                            timeline.append({
                                "timestamp": application.get("date"),
                                "type": "APPLICATION",
                                "target": application.get("target"),
                                "success_score": application.get("success_score"),
                                "notes": application.get("notes")
                            })
                    
                    # Add observation events
                    for observation in observations:
                        if observation.get("created"):
                            timeline.append({
                                "timestamp": observation.get("created"),
                                "type": "OBSERVATION_ADDED",
                                "observation_type": observation.get("type"),
                                "content_sample": self._truncate_content(observation.get("content", ""), 50)
                            })
                        
                        if observation.get("lastUpdated"):
                            timeline.append({
                                "timestamp": observation.get("lastUpdated"),
                                "type": "OBSERVATION_UPDATED",
                                "observation_type": observation.get("type"),
                                "content_sample": self._truncate_content(observation.get("content", ""), 50)
                            })
                    
                    # Sort timeline by timestamp
                    sorted_timeline = sorted(timeline, key=lambda t: t.get("timestamp", 0))
                    lesson_data["timeline"] = sorted_timeline
                    
                    results.append(lesson_data)
            
            return dict_to_json({"lessons": results})
                
        except Exception as e:
            error_msg = f"Error retrieving knowledge evolution: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def get_confidence_evolution(self, entity_name: str) -> str:
        """
        Track how confidence has evolved for a lesson over time.
        
        Args:
            entity_name: Name of the lesson entity
            
        Returns:
            JSON string with the confidence evolution data
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Get the lesson and its version history
            query = """
            MATCH (l:Entity {name: $entity_name, domain: 'lesson'})
            OPTIONAL MATCH path = (l)-[:SUPERSEDES*0..]->(old:Entity)
            WITH l, old, length(path) as distance
            RETURN l.name as lesson_name,
                   old.version as version,
                   old.created as timestamp,
                   old.confidence as confidence,
                   old.status as status
            ORDER BY old.created
            """
            
            records, _ = self.base_manager.safe_execute_query(
                query,
                {"entity_name": entity_name}
            )
            
            if not records or len(records) == 0:
                return dict_to_json({"error": f"Lesson '{entity_name}' not found"})
            
            # Process confidence data points
            confidence_data = []
            for record in records:
                timestamp = record.get("timestamp")
                confidence = record.get("confidence")
                version = record.get("version")
                status = record.get("status")
                
                if timestamp is not None and confidence is not None:
                    data_point = {
                        "timestamp": timestamp,
                        "confidence": confidence
                    }
                    
                    if version:
                        data_point["version"] = version
                    
                    if status:
                        data_point["status"] = status
                    
                    confidence_data.append(data_point)
            
            # Get application events that may have affected confidence
            application_query = """
            MATCH (l:Entity {name: $entity_name, domain: 'lesson'})-[r:APPLIED_TO]->(target)
            RETURN target.name as target,
                   r.applied_date as timestamp,
                   r.success_score as success_score
            ORDER BY r.applied_date
            """
            
            application_records, _ = self.base_manager.safe_execute_query(
                application_query,
                {"entity_name": entity_name}
            )
            
            # Process application events
            application_events = []
            if application_records:
                for record in application_records:
                    timestamp = record.get("timestamp")
                    success_score = record.get("success_score")
                    target = record.get("target")
                    
                    if timestamp is not None:
                        event = {
                            "timestamp": timestamp,
                            "type": "APPLICATION",
                            "target": target
                        }
                        
                        if success_score is not None:
                            event["success_score"] = success_score
                        
                        application_events.append(event)
            
            # Calculate trends
            trend_data = self._calculate_confidence_trend(confidence_data)
            
            return dict_to_json({
                "lesson": entity_name,
                "confidence_history": confidence_data,
                "application_events": application_events,
                "trend": trend_data
            })
                
        except Exception as e:
            error_msg = f"Error retrieving confidence evolution: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def get_lesson_application_impact(self, entity_name: str) -> str:
        """
        Analyze the impact of lesson applications on success metrics.
        
        Args:
            entity_name: Name of the lesson entity
            
        Returns:
            JSON string with application impact analysis
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Get all application events for the lesson
            query = """
            MATCH (l:Entity {name: $entity_name, domain: 'lesson'})-[r:APPLIED_TO]->(target)
            RETURN target.name as target,
                   r.applied_date as timestamp,
                   r.success_score as success_score,
                   r.application_notes as notes
            ORDER BY r.applied_date
            """
            
            records, _ = self.base_manager.safe_execute_query(
                query,
                {"entity_name": entity_name}
            )
            
            if not records:
                return dict_to_json({
                    "lesson": entity_name,
                    "application_count": 0,
                    "message": "No application events found for this lesson"
                })
            
            # Process application data
            applications = []
            success_scores = []
            targets = set()
            
            for record in records:
                timestamp = record.get("timestamp")
                success_score = record.get("success_score")
                target = record.get("target")
                notes = record.get("notes")
                
                application = {
                    "timestamp": timestamp,
                    "target": target
                }
                
                if success_score is not None:
                    application["success_score"] = success_score
                    success_scores.append(success_score)
                
                if notes:
                    application["notes"] = notes
                
                applications.append(application)
                
                if target:
                    targets.add(target)
            
            # Calculate metrics
            application_count = len(applications)
            unique_target_count = len(targets)
            
            avg_success = None
            min_success = None
            max_success = None
            
            if success_scores:
                avg_success = sum(success_scores) / len(success_scores)
                min_success = min(success_scores)
                max_success = max(success_scores)
            
            # Calculate timeline stats
            if application_count >= 2:
                first_application = min(applications, key=lambda a: a.get("timestamp", 0))
                last_application = max(applications, key=lambda a: a.get("timestamp", 0))
                
                time_span = last_application.get("timestamp", 0) - first_application.get("timestamp", 0)
                days_span = time_span / (24 * 60 * 60) if time_span > 0 else 0
                
                timeline_stats = {
                    "first_application": first_application.get("timestamp"),
                    "last_application": last_application.get("timestamp"),
                    "days_span": round(days_span, 1),
                    "applications_per_day": round(application_count / days_span, 2) if days_span > 0 else 0
                }
            else:
                timeline_stats = {
                    "first_application": applications[0].get("timestamp") if applications else None,
                    "applications_per_day": 0
                }
            
            # Return analysis results
            result = {
                "lesson": entity_name,
                "application_count": application_count,
                "unique_target_count": unique_target_count,
                "timeline_stats": timeline_stats,
                "applications": applications
            }
            
            if success_scores:
                result["success_metrics"] = {
                    "average_success": avg_success,
                    "min_success": min_success,
                    "max_success": max_success
                }
            
            return dict_to_json(result)
                
        except Exception as e:
            error_msg = f"Error analyzing lesson application impact: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def get_learning_progression(self, entity_name: str, max_depth: int = 3) -> str:
        """
        Analyze the learning progression path for a lesson.
        
        Args:
            entity_name: Name of the lesson entity
            max_depth: Maximum depth for the progression graph
            
        Returns:
            JSON string with the learning progression graph
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Validate depth
            if max_depth < 1:
                max_depth = 1
            elif max_depth > 5:  # Cap maximum depth
                max_depth = 5
            
            # Get lesson relationships in both directions
            query = f"""
            MATCH (l:Entity {{name: $entity_name, domain: 'lesson'}})
            
            // Get BUILDS_ON chain (things this lesson builds on)
            OPTIONAL MATCH path1 = (l)-[:BUILDS_ON*1..{max_depth}]->(prerequisite:Entity)
            
            // Get lessons that build on this
            OPTIONAL MATCH path2 = (l)<-[:BUILDS_ON*1..{max_depth}]-(dependent:Entity)
            
            // Collect all nodes
            WITH l, 
                 collect(distinct prerequisite) as prerequisites,
                 collect(distinct dependent) as dependents
            
            // Return nodes and their relationships
            RETURN l as lesson,
                   prerequisites,
                   dependents
            """
            
            records, _ = self.base_manager.safe_execute_query(
                query,
                {"entity_name": entity_name}
            )
            
            if not records or len(records) == 0:
                return dict_to_json({"error": f"Lesson '{entity_name}' not found"})
            
            # Process results
            record = records[0]
            lesson = record.get("lesson")
            prerequisites = record.get("prerequisites", [])
            dependents = record.get("dependents", [])
            
            # Extract lesson data
            lesson_data = dict(lesson.items()) if lesson else {}
            
            # Extract prerequisite data
            prerequisite_data = []
            for prereq in prerequisites:
                if prereq:
                    prereq_dict = dict(prereq.items())
                    prerequisite_data.append(prereq_dict)
            
            # Extract dependent data
            dependent_data = []
            for dep in dependents:
                if dep:
                    dep_dict = dict(dep.items())
                    dependent_data.append(dep_dict)
            
            # Get relationship details
            relationship_query = f"""
            MATCH (l:Entity {{name: $entity_name, domain: 'lesson'}})
            
            // Get direct BUILDS_ON relationships
            OPTIONAL MATCH (l)-[r1:BUILDS_ON]->(direct:Entity)
            
            // Get direct relationships from dependents
            OPTIONAL MATCH (l)<-[r2:BUILDS_ON]-(direct_dep:Entity)
            
            // Return relationship data
            RETURN collect({{from: l.name, to: direct.name, type: 'BUILDS_ON', properties: properties(r1)}}) as outgoing,
                   collect({{from: direct_dep.name, to: l.name, type: 'BUILDS_ON', properties: properties(r2)}}) as incoming
            """
            
            relationship_records, _ = self.base_manager.safe_execute_query(
                relationship_query,
                {"entity_name": entity_name}
            )
            
            relationships = []
            if relationship_records and len(relationship_records) > 0:
                outgoing = relationship_records[0].get("outgoing", [])
                incoming = relationship_records[0].get("incoming", [])
                
                for rel in outgoing + incoming:
                    if rel.get("from") and rel.get("to"):
                        relationships.append(rel)
            
            # Create network graph
            nodes = []
            
            # Add central lesson
            if lesson_data:
                lesson_data["role"] = "central"
                nodes.append(lesson_data)
            
            # Add prerequisites
            for prereq in prerequisite_data:
                prereq["role"] = "prerequisite"
                nodes.append(prereq)
            
            # Add dependents
            for dep in dependent_data:
                dep["role"] = "dependent"
                nodes.append(dep)
            
            # Return progression graph
            return dict_to_json({
                "lesson": entity_name,
                "nodes": nodes,
                "relationships": relationships,
                "prerequisite_count": len(prerequisite_data),
                "dependent_count": len(dependent_data)
            })
                
        except Exception as e:
            error_msg = f"Error analyzing learning progression: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def _truncate_content(self, content: str, max_length: int = 50) -> str:
        """Truncate content to specified length with ellipsis."""
        if not content:
            return ""
        
        if len(content) <= max_length:
            return content
        
        return content[:max_length] + "..."
    
    def _calculate_confidence_trend(self, confidence_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trend information from confidence history."""
        if not confidence_data or len(confidence_data) < 2:
            return {"trend": "static", "change": 0.0}
        
        # Get first and last confidence values
        first_point = confidence_data[0]
        last_point = confidence_data[-1]
        
        first_confidence = first_point.get("confidence", 0.0)
        last_confidence = last_point.get("confidence", 0.0)
        
        # Calculate change
        confidence_change = last_confidence - first_confidence
        
        # Determine trend
        if abs(confidence_change) < 0.05:  # Less than 5% change
            trend = "static"
        elif confidence_change > 0:
            if confidence_change > 0.2:  # More than 20% increase
                trend = "strong_increase"
            else:
                trend = "increase"
        else:
            if confidence_change < -0.2:  # More than 20% decrease
                trend = "strong_decrease"
            else:
                trend = "decrease"
        
        return {
            "trend": trend,
            "change": confidence_change,
            "initial": first_confidence,
            "current": last_confidence,
            "time_period": last_point.get("timestamp", 0) - first_point.get("timestamp", 0)
        } 