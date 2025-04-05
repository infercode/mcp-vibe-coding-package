from typing import Any, Dict, List, Optional, Union
import time
import json
import logging
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
        self.logger = logging.getLogger(__name__)
    
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
            
            # Execute query using safe_execute_read_query (read-only operation)
            records = self.base_manager.safe_execute_read_query(
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
                    timeline.sort(key=lambda x: x.get("timestamp", 0))
                    
                    # Add to lesson data
                    lesson_data["timeline"] = timeline
                    
                    results.append(lesson_data)
            
            # Format the final response
            return dict_to_json({
                "lessons": results,
                "count": len(results),
                "filters": {
                    "entity_name": entity_name,
                    "lesson_type": lesson_type,
                    "start_date": start_date,
                    "end_date": end_date,
                    "include_superseded": include_superseded
                }
            })
                
        except Exception as e:
            error_msg = f"Error retrieving knowledge evolution: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def get_confidence_evolution(self, entity_name: str) -> str:
        """
        Track how confidence in knowledge has changed over time.
        
        Args:
            entity_name: Name of the entity to track
            
        Returns:
            JSON string with confidence evolution data
        """
        try:
            self.base_manager.ensure_initialized()
            
            # First verify entity exists
            entity_query = """
            MATCH (e:Entity {name: $name})
            WHERE e.domain = 'lesson'
            RETURN e
            """
            
            # Execute query using safe_execute_read_query (read-only operation)
            entity_records = self.base_manager.safe_execute_read_query(
                entity_query,
                {"name": entity_name}
            )
            
            if not entity_records or len(entity_records) == 0:
                return dict_to_json({"error": f"Entity '{entity_name}' not found"})
            
            # Get the entity and all previous versions
            query = """
            MATCH (e:Entity {name: $name})
            WHERE e.domain = 'lesson'
            OPTIONAL MATCH path = (e)-[:SUPERSEDES*0..]->(old:Entity)
            RETURN old.id as id,
                   old.name as name,
                   old.version as version,
                   old.created as created,
                   old.confidence as confidence,
                   old.status as status,
                   length(path) as distance
            ORDER BY old.created DESC
            """
            
            # Execute query using safe_execute_read_query (read-only operation)
            records = self.base_manager.safe_execute_read_query(
                query,
                {"name": entity_name}
            )
            
            if not records:
                return dict_to_json({
                    "entity": entity_name,
                    "message": "No version history found",
                    "confidence_data": []
                })
            
            # Process confidence data
            confidence_data = []
            for record in records:
                confidence = record.get("confidence")
                if confidence is not None:
                    confidence_data.append({
                        "id": record.get("id"),
                        "version": record.get("version"),
                        "timestamp": record.get("created"),
                        "confidence": confidence,
                        "status": record.get("status")
                    })
            
            # Sort by timestamp
            confidence_data.sort(key=lambda x: x.get("timestamp", 0))
            
            # Calculate confidence trend
            trend_data = self._calculate_confidence_trend(confidence_data)
            
            # Format response
            return dict_to_json({
                "entity": entity_name,
                "confidence_data": confidence_data,
                "trend": trend_data
            })
                
        except Exception as e:
            error_msg = f"Error retrieving confidence evolution: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def get_lesson_application_impact(self, entity_name: str) -> str:
        """
        Analyze the impact of lesson application over time.
        
        Args:
            entity_name: Name of the entity to analyze
            
        Returns:
            JSON string with application impact data
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Verify entity exists
            entity_query = """
            MATCH (e:Entity {name: $name})
            WHERE e.domain = 'lesson'
            RETURN e
            """
            
            # Execute query using safe_execute_read_query (read-only operation)
            entity_records = self.base_manager.safe_execute_read_query(
                entity_query,
                {"name": entity_name}
            )
            
            if not entity_records or len(entity_records) == 0:
                return dict_to_json({"error": f"Entity '{entity_name}' not found"})
            
            # Get application data
            query = """
            MATCH (e:Entity {name: $name})-[r:APPLIED_TO]->(target)
            RETURN target.name as target,
                   target.entityType as target_type,
                   r.applied_date as date,
                   r.success_score as success_score,
                   r.application_notes as notes,
                   r.properties as properties
            ORDER BY r.applied_date DESC
            """
            
            # Execute query using safe_execute_read_query (read-only operation)
            records = self.base_manager.safe_execute_read_query(
                query,
                {"name": entity_name}
            )
            
            if not records:
                return dict_to_json({
                    "entity": entity_name,
                    "message": "No application history found",
                    "applications": []
                })
            
            # Process application data
            applications = []
            for record in records:
                applications.append({
                    "target": record.get("target"),
                    "target_type": record.get("target_type"),
                    "date": record.get("date"),
                    "success_score": record.get("success_score"),
                    "notes": record.get("notes"),
                    "properties": record.get("properties", {})
                })
            
            # Calculate statistics
            application_count = len(applications)
            success_scores = [app.get("success_score", 0) for app in applications if app.get("success_score") is not None]
            
            avg_success = sum(success_scores) / len(success_scores) if success_scores else 0
            
            # Group by target type
            target_types = {}
            for app in applications:
                target_type = app.get("target_type", "unknown")
                if target_type not in target_types:
                    target_types[target_type] = []
                target_types[target_type].append(app)
            
            # Calculate success trends over time
            success_trend = []
            if applications:
                # Sort by date
                sorted_apps = sorted(applications, key=lambda x: x.get("date", 0))
                
                # Group by month/year
                time_periods = {}
                for app in sorted_apps:
                    date = app.get("date")
                    if date:
                        dt = datetime.fromtimestamp(date)
                        period = f"{dt.year}-{dt.month:02d}"
                        
                        if period not in time_periods:
                            time_periods[period] = []
                        
                        time_periods[period].append(app)
                
                # Calculate average success by period
                for period, period_apps in time_periods.items():
                    period_scores = [app.get("success_score", 0) for app in period_apps if app.get("success_score") is not None]
                    period_avg = sum(period_scores) / len(period_scores) if period_scores else 0
                    
                    success_trend.append({
                        "period": period,
                        "count": len(period_apps),
                        "avg_success": period_avg
                    })
            
            # Format response
            return dict_to_json({
                "entity": entity_name,
                "application_count": application_count,
                "average_success": avg_success,
                "target_type_distribution": {
                    type_name: len(apps) for type_name, apps in target_types.items()
                },
                "success_trend": success_trend,
                "applications": applications
            })
                
        except Exception as e:
            error_msg = f"Error analyzing lesson application impact: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def get_learning_progression(self, entity_name: str, max_depth: int = 3) -> str:
        """
        Analyze the learning progression by tracking superseded versions.
        
        Args:
            entity_name: Name of the entity to analyze
            max_depth: Maximum depth of superseded relationships to traverse
            
        Returns:
            JSON string with learning progression data
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Validate max_depth
            if max_depth < 1:
                max_depth = 1
            elif max_depth > 10:
                max_depth = 10
            
            # First get the entity
            entity_query = """
            MATCH (e:Entity {name: $name})
            WHERE e.domain = 'lesson'
            RETURN e
            """
            
            # Execute query using safe_execute_read_query (read-only operation)
            entity_records = self.base_manager.safe_execute_read_query(
                entity_query,
                {"name": entity_name}
            )
            
            if not entity_records or len(entity_records) == 0:
                return dict_to_json({"error": f"Entity '{entity_name}' not found"})
            
            # Get the progression chain both forward and backward
            query = f"""
            MATCH (e:Entity {{name: $name}})
            WHERE e.domain = 'lesson'
            OPTIONAL MATCH forward_path = (e)-[:SUPERSEDES*1..{max_depth}]->(old:Entity)
            OPTIONAL MATCH backward_path = (new:Entity)-[:SUPERSEDES*1..{max_depth}]->(e)
            WITH e, forward_path, backward_path, old, new
            RETURN DISTINCT e.id as id,
                   e.name as name,
                   e.created as created,
                   e.confidence as confidence,
                   e.entityType as type,
                   e.status as status,
                   COLLECT(DISTINCT {{
                     id: old.id,
                     name: old.name,
                     created: old.created,
                     confidence: old.confidence,
                     status: old.status,
                     relationship: 'SUPERSEDED_BY',
                     direction: 'old'
                   }}) as older_versions,
                   COLLECT(DISTINCT {{
                     id: new.id,
                     name: new.name,
                     created: new.created,
                     confidence: new.confidence,
                     status: new.status,
                     relationship: 'SUPERSEDES',
                     direction: 'new'
                   }}) as newer_versions
            """
            
            # Execute query using safe_execute_read_query (read-only operation)
            records = self.base_manager.safe_execute_read_query(
                query,
                {"name": entity_name}
            )
            
            if not records:
                return dict_to_json({
                    "entity": entity_name,
                    "message": "No progression data found",
                    "progression": []
                })
            
            # Process progression data
            entity_data = {}
            older_versions = []
            newer_versions = []
            
            for record in records:
                # Core entity
                entity_data = {
                    "id": record.get("id"),
                    "name": record.get("name"),
                    "created": record.get("created"),
                    "confidence": record.get("confidence"),
                    "type": record.get("type"),
                    "status": record.get("status"),
                    "is_current": True
                }
                
                # Process older versions
                record_older = record.get("older_versions", [])
                for version in record_older:
                    if version.get("id"):  # Ensure empty nodes are filtered out
                        older_versions.append(version)
                
                # Process newer versions
                record_newer = record.get("newer_versions", [])
                for version in record_newer:
                    if version.get("id"):  # Ensure empty nodes are filtered out
                        newer_versions.append(version)
            
            # Sort by created date
            older_versions.sort(key=lambda x: x.get("created", 0), reverse=True)
            newer_versions.sort(key=lambda x: x.get("created", 0))
            
            # Build the complete progression chain
            progression = []
            
            # Add newer versions (from newest to target)
            for version in newer_versions:
                version["is_current"] = False
                progression.append(version)
            
            # Add current version
            progression.append(entity_data)
            
            # Add older versions (from target to oldest)
            for version in older_versions:
                version["is_current"] = False
                progression.append(version)
            
            # Get improvement metrics
            confidence_values = [v.get("confidence", 0) for v in progression if v.get("confidence") is not None]
            confidence_change = 0
            confidence_growth = 0
            
            if len(confidence_values) >= 2:
                first_confidence = confidence_values[-1]  # Oldest version
                current_confidence = entity_data.get("confidence", 0)
                confidence_change = current_confidence - first_confidence
                confidence_growth = (current_confidence / first_confidence - 1) * 100 if first_confidence > 0 else 0
            
            # Format response
            return dict_to_json({
                "entity": entity_name,
                "current": entity_data,
                "progression": progression,
                "version_count": len(progression),
                "newer_versions": len(newer_versions),
                "older_versions": len(older_versions),
                "metrics": {
                    "confidence_change": confidence_change,
                    "confidence_growth_percentage": confidence_growth,
                    "progression_duration_days": self._calculate_duration_days(progression)
                }
            })
                
        except Exception as e:
            error_msg = f"Error analyzing learning progression: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def _truncate_content(self, content: str, max_length: int = 50) -> str:
        """
        Truncate content to specified length.
        
        Args:
            content: Text content to truncate
            max_length: Maximum length
            
        Returns:
            Truncated content with ellipsis if needed
        """
        if not content or len(content) <= max_length:
            return content
        return content[:max_length] + "..."
    
    def _calculate_confidence_trend(self, confidence_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate trend metrics for confidence evolution.
        
        Args:
            confidence_data: List of confidence data points
            
        Returns:
            Dict with trend metrics
        """
        if not confidence_data or len(confidence_data) < 2:
            return {
                "direction": "stable",
                "change": 0,
                "growth_percentage": 0
            }
        
        # Extract confidence values
        confidence_values = [d.get("confidence", 0) for d in confidence_data if d.get("confidence") is not None]
        
        if not confidence_values or len(confidence_values) < 2:
            return {
                "direction": "stable",
                "change": 0,
                "growth_percentage": 0
            }
        
        # Calculate change
        first_confidence = confidence_values[0]
        last_confidence = confidence_values[-1]
        change = last_confidence - first_confidence
        
        # Calculate growth percentage
        growth_percentage = (last_confidence / first_confidence - 1) * 100 if first_confidence > 0 else 0
        
        # Determine direction
        if change > 0.05:
            direction = "increasing"
        elif change < -0.05:
            direction = "decreasing"
        else:
            direction = "stable"
        
        return {
            "direction": direction,
            "change": change,
            "growth_percentage": growth_percentage,
            "initial_confidence": first_confidence,
            "current_confidence": last_confidence,
            "data_points": len(confidence_values)
        }
    
    def _calculate_duration_days(self, progression: List[Dict[str, Any]]) -> float:
        """
        Calculate the duration of a progression in days.
        
        Args:
            progression: List of versions in the progression
            
        Returns:
            Duration in days
        """
        if not progression or len(progression) < 2:
            return 0
        
        # Find oldest and newest timestamps
        timestamps = [v.get("created", 0) for v in progression if v.get("created") is not None]
        
        if not timestamps or len(timestamps) < 2:
            return 0
        
        oldest = min(timestamps)
        newest = max(timestamps)
        
        # Calculate duration in days
        duration_seconds = newest - oldest
        duration_days = duration_seconds / (24 * 60 * 60)
        
        return duration_days 