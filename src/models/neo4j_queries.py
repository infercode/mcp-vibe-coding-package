#!/usr/bin/env python3
"""
Neo4j Query Parameter Models

This module provides Pydantic models for Neo4j query parameters and results,
enabling type-safe query execution and result handling.
"""

from typing import Dict, List, Optional, Any, Union, Set, Literal, ClassVar, Annotated
from datetime import datetime
from pydantic import (
    BaseModel, 
    Field, 
    field_validator, 
    model_validator, 
    ConfigDict,
    computed_field,
    model_serializer
)
import re

# --- Neo4j Parameter Types ---

class CypherString(BaseModel):
    """Model for Cypher string parameters with validation."""
    value: Annotated[str, Field(description="String value for the Cypher query")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    @field_validator('value')
    @classmethod
    def validate_string(cls, v: str) -> str:
        """Validate the string to prevent injection attacks."""
        # Check for null bytes which can be used for injection
        if '\0' in v:
            raise ValueError("String contains null bytes, which is not allowed")
        return v
    
    @model_serializer
    def serialize_model(self) -> str:
        """Convert to a form suitable for Neo4j parameters."""
        return self.value


class CypherNumber(BaseModel):
    """Model for numeric parameters in Cypher queries."""
    value: Annotated[Union[int, float], Field(description="Numeric value for the Cypher query")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    @model_serializer
    def serialize_model(self) -> Union[int, str]:
        """Convert to a form suitable for Neo4j parameters."""
        # For float types, convert to string to avoid Neo4j driver issues
        if isinstance(self.value, float):
            return str(self.value)
        return self.value


class CypherBoolean(BaseModel):
    """Model for boolean parameters in Cypher queries."""
    value: Annotated[bool, Field(description="Boolean value for the Cypher query")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    @model_serializer
    def serialize_model(self) -> bool:
        """Convert to a form suitable for Neo4j parameters."""
        return self.value


class CypherList(BaseModel):
    """Model for list parameters in Cypher queries."""
    values: Annotated[List[Any], Field(description="List of values for the Cypher query")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    @field_validator('values')
    @classmethod
    def validate_list(cls, v: List[Any]) -> List[Any]:
        """Validate list items are of supported types."""
        for item in v:
            if not isinstance(item, (str, int, float, bool)):
                raise ValueError(f"Unsupported type in list: {type(item)}")
        return v
    
    @model_serializer
    def serialize_model(self) -> List[Any]:
        """Convert to a form suitable for Neo4j parameters."""
        return self.values


class CypherDict(BaseModel):
    """Model for dictionary parameters in Cypher queries."""
    values: Annotated[Dict[str, Any], Field(description="Dictionary of values for the Cypher query")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    @field_validator('values')
    @classmethod
    def validate_dict(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate dictionary values are of supported types."""
        for key, value in v.items():
            if not isinstance(key, str):
                raise ValueError(f"Dictionary keys must be strings, got: {type(key)}")
            if not isinstance(value, (str, int, float, bool, list, dict)):
                raise ValueError(f"Unsupported type in dictionary: {type(value)}")
        return v
    
    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Convert to a form suitable for Neo4j parameters."""
        return self.values


# --- Query Parameter Models ---

class NodeProperties(BaseModel):
    """Model for node properties in Cypher queries."""
    name: Optional[str] = Field(None, description="Name property value")
    type: Optional[str] = Field(None, description="Type property value")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    tags: Optional[List[str]] = Field(None, description="List of tags")
    additional_properties: Optional[Dict[str, Any]] = Field(None, description="Additional properties")
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    def to_cypher_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary suitable for Cypher queries."""
        result = {}
        for field, value in self:
            if field == "additional_properties" and value:
                # Add additional properties to the result
                result.update(value)
            elif value is not None:
                result[field] = value
        return result


class RelationshipProperties(BaseModel):
    """Model for relationship properties in Cypher queries."""
    type: Optional[str] = Field(None, description="Relationship type")
    weight: Annotated[Optional[float], Field(None, ge=0.0, le=1.0, description="Relationship weight")]
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    confidence: Annotated[Optional[float], Field(None, ge=0.0, le=1.0, description="Confidence score")]
    additional_properties: Optional[Dict[str, Any]] = Field(None, description="Additional properties")
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    def to_cypher_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary suitable for Cypher queries."""
        result = {}
        for field, value in self:
            if field == "additional_properties" and value:
                # Add additional properties to the result
                result.update(value)
            elif value is not None:
                result[field] = value
        return result


class QueryLimit(BaseModel):
    """Model for query limits with validation."""
    value: Annotated[int, Field(ge=1, le=1000, description="Maximum number of results to return")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    def __int__(self) -> int:
        """Allow direct conversion to int."""
        return self.value


class QuerySkip(BaseModel):
    """Model for query skip with validation."""
    value: Annotated[int, Field(ge=0, description="Number of results to skip")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    def __int__(self) -> int:
        """Allow direct conversion to int."""
        return self.value


# --- Cypher Query Models ---

class CypherParameters(BaseModel):
    """Model for Cypher query parameters with validation."""
    parameters: Annotated[Dict[str, Any], Field(default_factory=dict, description="Parameters for the Cypher query")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    @field_validator('parameters')
    @classmethod
    def validate_parameters(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameter values are of supported types."""
        for key, value in v.items():
            # Check for valid parameter name
            if not key.isidentifier():
                raise ValueError(f"Invalid parameter name: {key}. Must be a valid identifier.")
            
            # Validate parameter value types
            if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                raise ValueError(f"Unsupported type for parameter {key}: {type(value)}")
        
        return v
    
    def to_neo4j_parameters(self) -> Dict[str, Any]:
        """Convert to Neo4j driver compatible parameters."""
        result = {}
        for key, value in self.parameters.items():
            # Handle special parameter conversions
            if isinstance(value, float) and not isinstance(value, bool):
                # Convert floats to strings for Neo4j driver
                result[key] = str(value)
            elif isinstance(value, (list, tuple)) and all(isinstance(x, (int, float, str, bool)) for x in value):
                # Lists of primitive types can be passed directly
                result[key] = list(value)
            elif isinstance(value, dict):
                # Convert dictionary values
                result[key] = {k: str(v) if isinstance(v, float) else v for k, v in value.items()}
            else:
                # Pass through other types
                result[key] = value
        
        return result


class CypherQuery(BaseModel):
    """Model for Cypher queries with validation and sanitization."""
    query: Annotated[str, Field(description="Cypher query string")]
    parameters: Optional[CypherParameters] = Field(None, description="Query parameters")
    database: Optional[str] = Field(None, description="Database to execute the query against")
    read_only: Annotated[bool, Field(default=True, description="Whether the query is read-only (non-destructive)")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    # Define destructive operations as class variables instead of private attributes
    destructive_operations: ClassVar[Set[str]] = {
        "CREATE", "DELETE", "REMOVE", "SET", "MERGE", "DROP",
        "CALL DB.INDEX", "CALL APOC", "WITH [0] AS FOO", "LOAD CSV",
        "PERIODIC COMMIT", "FOREACH"
    }
    
    # Define allowed query starters for read-only queries as class variables
    readonly_starters: ClassVar[Set[str]] = {
        "MATCH", "OPTIONAL MATCH", "WITH", "RETURN", "UNWIND", "WHERE",
        "CALL {", "CALL"  # Careful with CALL, need more specific checks
    }
    
    @model_validator(mode='after')
    @classmethod
    def validate_query(cls, values):
        """Validate the query string for security and correctness."""
        query = values.query
        read_only = values.read_only
        
        # Check if query is empty
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty")
        
        # Convert to uppercase for checks
        query_upper = query.upper()
        
        # Check for disallowed operations in read-only mode
        if read_only:
            # Use regex with word boundaries to check for destructive operations
            for operation in cls.destructive_operations:
                # Skip special multi-word operations
                if " " in operation:
                    if operation in query_upper:
                        raise ValueError(f"Destructive operation '{operation}' not allowed in read-only mode")
                    continue
                
                # Use word boundary check for single-word operations
                if re.search(r'\b' + re.escape(operation) + r'\b', query_upper):
                    raise ValueError(f"Destructive operation '{operation}' not allowed in read-only mode")
            
            # Ensure query starts with a read-only operation
            has_valid_starter = False
            for starter in cls.readonly_starters:
                if query_upper.lstrip().startswith(starter):
                    has_valid_starter = True
                    break
                    
            if not has_valid_starter:
                raise ValueError("Read-only query must start with a valid read operation (MATCH, RETURN, etc.)")
                
            # Special checks for CALL
            if "CALL" in query_upper and not ("CALL {" in query_upper):
                # Check for disallowed procedure calls
                disallowed_calls = ["CALL DB.", "CALL APOC.CREATE", "CALL APOC.DELETE", "CALL APOC.MERGE"]
                for call in disallowed_calls:
                    if call.upper() in query_upper:
                        raise ValueError(f"Procedure call '{call}' not allowed in read-only mode")
        
        # Set default parameters if none provided
        if values.parameters is None:
            values.parameters = CypherParameters(parameters={})
            
        return values
    
    @computed_field
    def query_type(self) -> str:
        """Determine if this is a read or write query."""
        return "read" if self.read_only else "write"
    
    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Convert to a form that can be executed by the Neo4j driver."""
        params = self.parameters.to_neo4j_parameters() if self.parameters else {}
        
        result = {
            "query": self.query,
            "parameters": params,
            "query_type": self.query_type
        }
        
        # Only include database if it's specified
        if self.database is not None:
            result["database"] = self.database
            
        return result


# --- Result Models ---

class NodeResult(BaseModel):
    """Model for a Neo4j node result."""
    id: Optional[str] = Field(None, description="Node ID")
    labels: Annotated[List[str], Field(default_factory=list, description="Node labels")]
    properties: Annotated[Dict[str, Any], Field(default_factory=dict, description="Node properties")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    @classmethod
    def from_neo4j_node(cls, node) -> 'NodeResult':
        """Create a NodeResult from a Neo4j Node object."""
        if hasattr(node, "id"):
            id = str(node.id)
        else:
            id = None
            
        if hasattr(node, "labels"):
            labels = list(node.labels)
        else:
            labels = []
            
        # Get properties from various sources
        properties = {}
        if hasattr(node, "items"):
            properties = dict(node.items())
        elif isinstance(node, dict):
            properties = node
        
        return cls(
            id=id,
            labels=labels,
            properties=properties
        )
    
    @computed_field
    def primary_label(self) -> Optional[str]:
        """Get the primary label of the node."""
        return self.labels[0] if self.labels else None


class RelationshipResult(BaseModel):
    """Model for a Neo4j relationship result."""
    id: Optional[str] = Field(None, description="Relationship ID")
    type: Annotated[str, Field(description="Relationship type")]
    properties: Annotated[Dict[str, Any], Field(default_factory=dict, description="Relationship properties")]
    start_node_id: Annotated[str, Field(description="ID of the start node")]
    end_node_id: Annotated[str, Field(description="ID of the end node")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    @classmethod
    def from_neo4j_relationship(cls, rel) -> 'RelationshipResult':
        """Create a RelationshipResult from a Neo4j Relationship object."""
        # Initialize default values
        id_val = None
        type_val = "RELATED_TO"
        properties = {}
        start_node_id = "unknown"  # Default fallback
        end_node_id = "unknown"    # Default fallback
        
        # Process based on object type
        if isinstance(rel, dict):
            # Handle dictionary representation
            id_val = str(rel.get("id", "unknown"))
            type_val = rel.get("type", "RELATED_TO")
            properties = rel.get("properties", {})
            start_node_id = str(rel.get("start_node_id", "unknown"))
            end_node_id = str(rel.get("end_node_id", "unknown"))
        else:
            # Handle Neo4j object representation
            # Extract ID
            if hasattr(rel, "id"):
                try:
                    id_val = str(rel.id)
                except (AttributeError, TypeError):
                    pass
                
            # Extract type
            if hasattr(rel, "type"):
                try:
                    type_val = rel.type
                except (AttributeError, TypeError):
                    pass
                
            # Extract properties
            if hasattr(rel, "items"):
                try:
                    properties = dict(rel.items())
                except (AttributeError, TypeError):
                    pass
            
            # Extract start node ID
            if hasattr(rel, "start_node"):
                try:
                    if hasattr(rel.start_node, "id"):
                        start_node_id = str(rel.start_node.id)
                except (AttributeError, TypeError):
                    pass
                    
            # Extract end node ID
            if hasattr(rel, "end_node"):
                try:
                    if hasattr(rel.end_node, "id"):
                        end_node_id = str(rel.end_node.id)
                except (AttributeError, TypeError):
                    pass
        
        return cls(
            id=id_val,
            type=type_val,
            properties=properties,
            start_node_id=start_node_id,
            end_node_id=end_node_id
        )
    
    @computed_field
    def relationship_label(self) -> str:
        """Generate a human-readable label for this relationship."""
        return f"{self.start_node_id} --[{self.type}]--> {self.end_node_id}"


class PathResult(BaseModel):
    """Model for a Neo4j path result."""
    nodes: Annotated[List[NodeResult], Field(default_factory=list, description="Nodes in the path")]
    relationships: Annotated[List[RelationshipResult], Field(default_factory=list, description="Relationships in the path")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    @classmethod
    def from_neo4j_path(cls, path) -> 'PathResult':
        """Create a PathResult from a Neo4j Path object."""
        nodes = []
        relationships = []
        
        if hasattr(path, "nodes"):
            nodes = [NodeResult.from_neo4j_node(node) for node in path.nodes]
        
        if hasattr(path, "relationships"):
            relationships = [RelationshipResult.from_neo4j_relationship(rel) for rel in path.relationships]
        
        return cls(
            nodes=nodes,
            relationships=relationships
        )
    
    @computed_field
    def path_length(self) -> int:
        """Get the length of the path (number of relationships)."""
        return len(self.relationships)


class QueryResult(BaseModel):
    """Model for a Neo4j query result."""
    records: Annotated[List[Dict[str, Any]], Field(default_factory=list, description="Query result records")]
    summary: Annotated[Dict[str, Any], Field(default_factory=dict, description="Query result summary")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    @classmethod
    def from_neo4j_result(cls, records: List[Dict[str, Any]], summary: Dict[str, Any]) -> 'QueryResult':
        """Create a QueryResult from Neo4j records and summary."""
        # Process records to handle Neo4j objects
        processed_records = []
        for record in records:
            processed_record = {}
            for key, value in record.items():
                # Handle different Neo4j types
                if hasattr(value, "labels"):  # Node
                    processed_record[key] = NodeResult.from_neo4j_node(value).model_dump()
                elif hasattr(value, "type") and hasattr(value, "start_node"):  # Relationship
                    processed_record[key] = RelationshipResult.from_neo4j_relationship(value).model_dump()
                elif hasattr(value, "nodes") and hasattr(value, "relationships"):  # Path
                    processed_record[key] = PathResult.from_neo4j_path(value).model_dump()
                else:
                    processed_record[key] = value
            processed_records.append(processed_record)
        
        return cls(
            records=processed_records,
            summary=summary
        )
    
    @computed_field
    def record_count(self) -> int:
        """Get the number of records in the result."""
        return len(self.records)


# --- Query Builder Models ---

class QueryOrder(BaseModel):
    """Model for ordering clauses in Cypher queries."""
    field: Annotated[str, Field(description="Field to order by")]
    direction: Annotated[Literal["ASC", "DESC"], Field(default="ASC", description="Order direction")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    def to_cypher(self) -> str:
        """Convert to Cypher ORDER BY clause."""
        return f"{self.field} {self.direction}"


class NodePattern(BaseModel):
    """Model for node patterns in Cypher queries."""
    variable: Annotated[str, Field(description="Variable name for the node")]
    labels: Optional[List[str]] = Field(None, description="Node labels")
    properties: Optional[Dict[str, Any]] = Field(None, description="Node properties")
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    @field_validator('variable')
    @classmethod
    def validate_variable(cls, v: str) -> str:
        """Validate variable is a valid Cypher identifier."""
        if not v or not v.isidentifier():
            raise ValueError(f"Invalid variable name: {v}. Must be a valid identifier.")
        return v
    
    def to_cypher(self) -> str:
        """Convert to Cypher node pattern."""
        # Create the labels part
        labels_str = ""
        if self.labels:
            labels_str = "".join(f":{label}" for label in self.labels)
        
        # Create the properties part
        props_str = ""
        if self.properties:
            props_parts = []
            for key, value in self.properties.items():
                if isinstance(value, str):
                    props_parts.append(f"{key}: '{value}'")
                else:
                    props_parts.append(f"{key}: {value}")
            
            if props_parts:
                props_str = f" {{ {', '.join(props_parts)} }}"
        
        return f"({self.variable}{labels_str}{props_str})"


class RelationshipPattern(BaseModel):
    """Model for relationship patterns in Cypher queries."""
    variable: Optional[str] = Field(None, description="Variable name for the relationship")
    type: Optional[str] = Field(None, description="Relationship type")
    properties: Optional[Dict[str, Any]] = Field(None, description="Relationship properties")
    direction: Annotated[Literal["OUTGOING", "INCOMING", "BOTH"], Field(default="OUTGOING", description="Relationship direction")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    @field_validator('variable')
    @classmethod
    def validate_variable(cls, v: Optional[str]) -> Optional[str]:
        """Validate variable is a valid Cypher identifier if provided."""
        if v and not v.isidentifier():
            raise ValueError(f"Invalid variable name: {v}. Must be a valid identifier.")
        return v
    
    def to_cypher(self) -> str:
        """Convert to Cypher relationship pattern."""
        # Create the type part
        type_str = ""
        if self.type:
            type_str = f":{self.type}"
        
        # Create the properties part
        props_str = ""
        if self.properties:
            props_parts = []
            for key, value in self.properties.items():
                if isinstance(value, str):
                    props_parts.append(f"{key}: '{value}'")
                else:
                    props_parts.append(f"{key}: {value}")
            
            if props_parts:
                props_str = f" {{ {', '.join(props_parts)} }}"
        
        # Create the variable part
        var_str = ""
        if self.variable:
            var_str = self.variable
        
        # Create the direction arrows
        if self.direction == "OUTGOING":
            return f"-[{var_str}{type_str}{props_str}]->"
        elif self.direction == "INCOMING":
            return f"<-[{var_str}{type_str}{props_str}]-"
        else:  # BOTH
            return f"-[{var_str}{type_str}{props_str}]-"


class PathPattern(BaseModel):
    """Model for path patterns in Cypher queries."""
    nodes: Annotated[List[NodePattern], Field(description="Nodes in the path")]
    relationships: Annotated[List[RelationshipPattern], Field(description="Relationships in the path")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    @model_validator(mode='after')
    @classmethod
    def validate_pattern(cls, values):
        """Validate the path pattern structure."""
        nodes = values.nodes
        relationships = values.relationships
        
        # Check if number of nodes is at least 2
        if len(nodes) < 2:
            raise ValueError("PathPattern must have at least 2 nodes")
        
        # Check if number of relationships matches number of nodes - 1
        if len(relationships) != len(nodes) - 1:
            raise ValueError(f"Number of relationships ({len(relationships)}) must be one less than number of nodes ({len(nodes)})")
        
        return values
    
    def to_cypher(self) -> str:
        """Convert to Cypher path pattern."""
        parts = []
        for i, node in enumerate(self.nodes):
            parts.append(node.to_cypher())
            if i < len(self.relationships):
                parts.append(self.relationships[i].to_cypher())
        
        return "".join(parts)
    
    @computed_field
    def path_length(self) -> int:
        """Get the length of the path (number of relationships)."""
        return len(self.relationships)


class QueryBuilder(BaseModel):
    """Model for building Cypher queries."""
    match_patterns: Annotated[List[Union[NodePattern, PathPattern]], Field(default_factory=list, description="Match patterns")]
    where_clauses: Annotated[List[str], Field(default_factory=list, description="Where clauses")]
    return_fields: Annotated[List[str], Field(default_factory=list, description="Return fields")]
    order_by: Optional[List[QueryOrder]] = Field(None, description="Order by clauses")
    limit: Annotated[Optional[int], Field(default=None, ge=1, le=1000, description="Limit")]
    skip: Annotated[Optional[int], Field(default=None, ge=0, description="Skip")]
    parameters: Annotated[Dict[str, Any], Field(default_factory=dict, description="Query parameters")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    @computed_field
    def has_filters(self) -> bool:
        """Check if the query has any filtering clauses."""
        return len(self.where_clauses) > 0
    
    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Convert to a dictionary representation with query info."""
        result = self.model_dump()
        result["has_filters"] = self.has_filters
        return result
    
    def to_cypher_query(self) -> CypherQuery:
        """Convert to a CypherQuery object."""
        query_parts = []
        
        # MATCH clause
        if self.match_patterns:
            match_strs = []
            for pattern in self.match_patterns:
                match_strs.append(pattern.to_cypher())
            
            query_parts.append(f"MATCH {', '.join(match_strs)}")
        
        # WHERE clause
        if self.where_clauses:
            query_parts.append(f"WHERE {' AND '.join(self.where_clauses)}")
        
        # RETURN clause
        if self.return_fields:
            query_parts.append(f"RETURN {', '.join(self.return_fields)}")
        else:
            # Default return all
            query_parts.append("RETURN *")
        
        # ORDER BY clause
        if self.order_by:
            order_strs = [order.to_cypher() for order in self.order_by]
            query_parts.append(f"ORDER BY {', '.join(order_strs)}")
        
        # LIMIT clause
        if self.limit is not None:
            query_parts.append(f"LIMIT {self.limit}")
        
        # SKIP clause
        if self.skip is not None:
            query_parts.append(f"SKIP {self.skip}")
        
        # Create the full query
        query = " ".join(query_parts)
        
        # Create the CypherParameters object
        cypher_params = CypherParameters(parameters=self.parameters)
        
        # Create and return the CypherQuery
        return CypherQuery(
            query=query,
            parameters=cypher_params,
            database=None,  # Add database parameter as None
            read_only=True  # QueryBuilder always creates read-only queries
        ) 