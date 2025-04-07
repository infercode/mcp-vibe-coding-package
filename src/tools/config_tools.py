#!/usr/bin/env python3
"""
Configuration Tools with Pydantic Integration

This module implements MCP tools for configuration management using 
Pydantic models for validation and serialization.
"""

import os
import json
import uuid
import copy
import datetime
import re
from typing import Dict, Any, Optional, List, Union

from src.logger import get_logger
from src.models.config.models import (
    UnifiedConfig, ConfigUpdate, Neo4jConfig, EmbeddingConfig,
    Neo4jStatus, EmbeddingStatus, MemoryStatus,
    create_error_response, create_success_response, model_to_json, model_to_dict
)

# Initialize logger
logger = get_logger()

def register_config_tools(server, get_config_manager):
    """Register configuration tools with the server."""
    
    @server.tool()
    async def get_memory_status(client_id: Optional[str] = None) -> str:
        """
        Get the current status of the memory system.
        
        Args:
            client_id: Optional client ID for identifying the connection
        
        Returns:
            JSON string with the memory system status
        """
        try:
            # Sanitize client_id if provided
            if client_id:
                client_id = str(client_id).strip()
                
                # Check for valid format (if client_id has a specific expected format)
                if len(client_id) > 100:  # Prevent excessively long IDs
                    logger.warn(f"Client ID too long, truncating: {client_id[:20]}...")
                    client_id = client_id[:100]
            
            # Get the client-specific graph manager
            graph_manager = get_config_manager(client_id)
            
            # Get basic information about the memory system
            config = graph_manager.get_current_config()
            
            # Get Neo4j connection info
            neo4j_connected = graph_manager.check_connection()
            
            # Build status object based on the models
            neo4j_status = Neo4jStatus(
                connected=neo4j_connected,
                address=graph_manager.neo4j_uri,
                version=graph_manager.neo4j_version if hasattr(graph_manager, 'neo4j_version') else None,
                message="Connected to Neo4j" if neo4j_connected else "Not connected to Neo4j",
                timestamp=datetime.datetime.now().isoformat()
            )
            
            embedding_status = EmbeddingStatus(
                available=graph_manager.embedding_enabled,
                provider=graph_manager.embedder_provider,
                model=graph_manager.embedding_model,
                dimensions=getattr(graph_manager, 'embedding_dimensions', 1536),
                message="Embeddings enabled" if graph_manager.embedding_enabled else "Embeddings disabled",
                timestamp=datetime.datetime.now().isoformat()
            )
            
            memory_status = MemoryStatus(
                operational=neo4j_connected,
                neo4j=neo4j_status,
                embedding=embedding_status,
                message="Memory system operational" if neo4j_connected else "Memory system not fully operational",
                timestamp=datetime.datetime.now().isoformat()
            )
            
            # Create a success response
            success_response = create_success_response(
                message="Successfully retrieved memory status",
                data=model_to_dict(memory_status)
            )
            return model_to_json(success_response)
            
        except Exception as e:
            logger.error(f"Error getting memory status: {str(e)}", exc_info=True)
            error_response = create_error_response(
                message=f"Error getting memory status: {str(e)}",
                code="status_retrieval_error"
            )
            return model_to_json(error_response)

    @server.tool()
    async def get_unified_config(project_name: Optional[str] = None, config_content: Optional[str] = None) -> str:
        """
        Retrieves and applies unified configuration file for a client's project.
        
        This function should typically be called once per session unless the configuration
        has changed or you need to switch to a different project.
        
        Args:
            project_name: Optional. Name of the project to configure 
                           (alphanumeric, underscores, and hyphens only)
            config_content: Optional. JSON string containing configuration settings.
                           If provided, this will be used instead of loading from storage.
                           Must include required fields like "projectName" and "llmSettings".
        
        Returns:
            JSON string containing:
            - status: "success" or "error"
            - message: Description of the operation result
            - config: Object containing the unified configuration
            - warnings: Array of any warnings generated during processing
        """
        try:
            # Initialize warning tracking
            config_warnings = []
            
            # Validate and sanitize project_name
            if project_name is not None:
                original_project_name = project_name
                project_name = str(project_name).strip()
                
                # Check if sanitization changed the project name
                if project_name != original_project_name:
                    config_warnings.append({
                        "warning": "project_name_sanitized",
                        "message": "Project name was sanitized by removing whitespace"
                    })
                
                # Validate project name format (alphanumeric characters, underscores, and hyphens only)
                if project_name and not re.match(r'^[a-zA-Z0-9_-]+$', project_name):
                    error_response = create_error_response(
                        message="Invalid project name format. Use only letters, numbers, underscores, and hyphens.",
                        code="invalid_project_name",
                        details={"project_name": project_name}
                    )
                    return model_to_json(error_response)
                
                # Limit project name length to prevent abuse
                if len(project_name) > 50:
                    original_length = len(project_name)
                    project_name = project_name[:50]
                    logger.warn(f"Project name too long ({original_length} chars), truncating to: {project_name}")
                    config_warnings.append({
                        "warning": "project_name_truncated",
                        "message": f"Project name truncated from {original_length} to 50 characters"
                    })
            
            # Define the default config file name based on project name
            file_name = "mcp_unified_config.json"
            if project_name:
                file_name = f"mcp_unified_config_{project_name}.json"
            
            # Check if config content was provided
            if config_content:
                # Basic sanitization
                if not isinstance(config_content, str):
                    try:
                        config_content = str(config_content)
                        config_warnings.append({
                            "warning": "config_content_converted",
                            "message": "Config content was not a string and was converted to string format"
                        })
                    except Exception as conversion_error:
                        error_response = create_error_response(
                            message=f"Failed to convert config_content to string: {str(conversion_error)}",
                            code="invalid_config_type",
                            details={"type": type(config_content).__name__}
                        )
                        return model_to_json(error_response)
                
                # Sanitize config_content - trim whitespace only from start/end
                original_content_length = len(config_content)
                config_content = config_content.strip()
                
                # Check if sanitization changed the content
                if len(config_content) != original_content_length:
                    config_warnings.append({
                        "warning": "config_content_trimmed",
                        "message": "Config content was trimmed to remove leading/trailing whitespace"
                    })
                
                # Check content size to prevent excessive processing and potential DoS
                if len(config_content) > 1048576:  # 1MB limit (1024*1024)
                    error_response = create_error_response(
                        message=f"Configuration content exceeds maximum size limit of 1MB (received {len(config_content)/1024:.1f}KB)",
                        code="config_size_exceeded",
                        details={"max_size_kb": 1024, "received_size_kb": len(config_content)/1024}
                    )
                    return model_to_json(error_response)
                
                # Pre-check JSON format before attempting to parse
                # This provides better error messages than json.loads exceptions
                if not (config_content.startswith('{') and config_content.endswith('}')):
                    error_response = create_error_response(
                        message="Invalid JSON format: configuration must be a JSON object starting with '{' and ending with '}'",
                        code="invalid_json_format",
                        details={"expected_format": "JSON object"}
                    )
                    return model_to_json(error_response)
                
                # Check for common JSON syntax issues
                bracket_count = 0
                for char in config_content:
                    if char == '{':
                        bracket_count += 1
                    elif char == '}':
                        bracket_count -= 1
                        if bracket_count < 0:
                            error_response = create_error_response(
                                message="Invalid JSON format: unbalanced braces (too many closing braces)",
                                code="invalid_json_syntax",
                                details={"issue": "unbalanced_braces"}
                            )
                            return model_to_json(error_response)
                
                if bracket_count != 0:
                    error_response = create_error_response(
                        message="Invalid JSON format: unbalanced braces (missing closing braces)",
                        code="invalid_json_syntax",
                        details={"issue": "unbalanced_braces"}
                    )
                    return model_to_json(error_response)
                
                # Parse the config content
                try:
                    config = json.loads(config_content)
                    
                    # Validate basic structure
                    if not isinstance(config, dict):
                        error_response = create_error_response(
                            message="Configuration must be a JSON object (dictionary), not a list or primitive value",
                            code="invalid_config_format",
                            details={"received_type": type(config).__name__}
                        )
                        return model_to_json(error_response)
                    
                    # Check for required fields
                    required_fields = ["project_name"]
                    missing_fields = [field for field in required_fields if field not in config]
                    
                    if missing_fields:
                        error_response = create_error_response(
                            message=f"Missing required field(s) in configuration: {', '.join(missing_fields)}",
                            code="missing_required_fields",
                            details={"missing_fields": missing_fields}
                        )
                        return model_to_json(error_response)
                    
                    # Validate project_name in config
                    config_project_name = str(config["project_name"]).strip()
                    if not config_project_name:
                        error_response = create_error_response(
                            message="Project name in configuration cannot be empty",
                            code="invalid_project_name"
                        )
                        return model_to_json(error_response)
                    
                    # If project_name was provided in both args and config, ensure they match
                    if project_name and config_project_name != project_name:
                        config_warnings.append({
                            "warning": "project_name_mismatch",
                            "message": f"Project name in arguments ({project_name}) doesn't match config ({config_project_name})",
                            "using": config_project_name
                        })
                    
                    # Extract and validate client_id if present
                    client_id = config.get("client_id", None)
                    if client_id:
                        if not isinstance(client_id, str):
                            try:
                                client_id = str(client_id)
                                config_warnings.append({
                                    "warning": "client_id_converted",
                                    "message": "Client ID was not a string and was converted"
                                })
                            except Exception:
                                error_response = create_error_response(
                                    message="Client ID must be a string or convertible to string",
                                    code="invalid_client_id"
                                )
                                return model_to_json(error_response)
                                
                        client_id = client_id.strip()
                        if len(client_id) > 100:  # Prevent excessively long IDs
                            original_length = len(client_id)
                            logger.warn(f"Client ID too long ({original_length} chars), truncating")
                            client_id = client_id[:100]
                            config_warnings.append({
                                "warning": "client_id_truncated",
                                "message": f"Client ID truncated from {original_length} to 100 characters"
                            })
                    
                    # Get the client-specific graph manager
                    graph_manager = get_config_manager(client_id)
                    
                    # Validate Neo4j config if present
                    if "neo4j" in config:
                        neo4j_config = config["neo4j"]
                        if not isinstance(neo4j_config, dict):
                            error_response = create_error_response(
                                message="Neo4j configuration must be an object",
                                code="invalid_neo4j_config",
                                details={"received_type": type(neo4j_config).__name__}
                            )
                            return model_to_json(error_response)
                            
                        # Check for required Neo4j fields
                        required_neo4j_fields = ["uri", "username", "password"]
                        missing_neo4j_fields = [field for field in required_neo4j_fields if field not in neo4j_config]
                        
                        if missing_neo4j_fields:
                            config_warnings.append({
                                "warning": "missing_neo4j_fields",
                                "message": f"Neo4j configuration missing recommended field(s): {', '.join(missing_neo4j_fields)}",
                                "missing_fields": missing_neo4j_fields
                            })
                            logger.warn(f"Neo4j configuration missing field(s): {', '.join(missing_neo4j_fields)}")
                        
                        # Validate URI format if provided
                        if "uri" in neo4j_config:
                            uri = str(neo4j_config["uri"])
                            if not (uri.startswith("bolt://") or uri.startswith("neo4j://") or uri.startswith("neo4j+s://")):
                                config_warnings.append({
                                    "warning": "invalid_neo4j_uri_format",
                                    "message": "Neo4j URI should start with 'bolt://', 'neo4j://' or 'neo4j+s://'"
                                })
                    
                    # Validate embedding config if present
                    if "embeddings" in config:
                        embedding_config = config["embeddings"]
                        if not isinstance(embedding_config, dict):
                            error_response = create_error_response(
                                message="Embedding configuration must be an object",
                                code="invalid_embedding_config",
                                details={"received_type": type(embedding_config).__name__}
                            )
                            return model_to_json(error_response)
                        
                        # Check for required embedding fields
                        if "provider" in embedding_config:
                            provider = embedding_config["provider"]
                            
                            # Validate API key presence for providers that require it
                            api_key_providers = ["openai", "azure", "cohere", "anthropic"]
                            if provider.lower() in api_key_providers and "api_key" not in embedding_config:
                                config_warnings.append({
                                    "warning": "missing_api_key",
                                    "message": f"Provider '{provider}' typically requires an API key, but none was provided"
                                })
                    
                    # Apply the configuration
                    result = graph_manager.apply_client_config(config)
                    
                    # Return success or error message
                    if result["status"] == "success":
                        # Get the current configuration for confirmation
                        current_config = graph_manager.get_current_config()
                        
                        # Mask sensitive information before returning
                        if "neo4j" in current_config:
                            if "password" in current_config["neo4j"]:
                                current_config["neo4j"]["password"] = "********"
                            if "username" in current_config["neo4j"]:
                                # Don't completely mask username, show first character
                                username = current_config["neo4j"]["username"]
                                if username and len(username) > 1:
                                    masked_username = username[0] + "*" * (len(username) - 1)
                                    current_config["neo4j"]["username"] = masked_username
                            
                        if "embeddings" in current_config:
                            if "api_key" in current_config["embeddings"]:
                                current_config["embeddings"]["api_key"] = "********"
                            # Redact any private embedding settings
                            for key in list(current_config["embeddings"].keys()):
                                if "secret" in key.lower() or "private" in key.lower() or "token" in key.lower():
                                    current_config["embeddings"][key] = "********"
                            
                        success_response = create_success_response(
                            message="Configuration applied successfully",
                            data={
                                "config": current_config,
                                "file_name": file_name,
                                "timestamp": datetime.datetime.now().isoformat()
                            }
                        )
                        
                        # Add any warnings collected during processing
                        if config_warnings:
                            success_response.data["warnings"] = config_warnings
                            
                        return model_to_json(success_response)
                    else:
                        error_response = create_error_response(
                            message=result.get("message", "Unknown error applying configuration"),
                            code="config_application_error",
                            details=result.get("details", {})
                        )
                        return model_to_json(error_response)
                        
                except json.JSONDecodeError as json_error:
                    # Provide detailed error information for JSON parsing issues
                    # Calculate excerpt around the error location
                    start_pos = max(0, json_error.pos - 20)
                    end_pos = min(len(config_content), json_error.pos + 20)
                    excerpt = config_content[start_pos:end_pos]
                    
                    # Add a marker to show the error position
                    position_in_excerpt = json_error.pos - start_pos
                    marked_excerpt = excerpt[:position_in_excerpt] + ">>>" + excerpt[position_in_excerpt:] 
                    
                    error_response = create_error_response(
                        message=f"Invalid JSON configuration: {str(json_error)}",
                        code="invalid_json_error",
                        details={
                            "error_position": json_error.pos,
                            "error_line": json_error.lineno,
                            "error_column": json_error.colno,
                            "excerpt": marked_excerpt
                        }
                    )
                    return model_to_json(error_response)
            
            # If no config content was provided, request the client to read the file
            # Format the response using direct instruction format
            command_message = (
                f"DIRECT INSTRUCTIONS FOR AI ASSISTANT - REQUIRED IMMEDIATE ACTION:\n\n"
                f"AI ASSISTANT: YOU MUST COMPLETE ALL STEPS IN ORDER:\n\n"
                f"STEP 1: CHECK IF CONFIGURATION FILE EXISTS\n"
                f"Filename: {file_name}\n"
                f"Location: Project root directory\n\n"
                f"STEP 2: IF FILE EXISTS - READ CONFIGURATION FILE\n"
                f"Read the content of the file and verify it is valid JSON\n\n"
                f"STEP 3: IF FILE EXISTS - CALL get_unified_config AGAIN WITH CONTENT\n"
                f"Call the get_unified_config tool with these parameters:\n"
                f"project_name: \"{project_name}\"\n"
                f"config_content: (REPLACE THIS WITH THE ACTUAL FILE CONTENT - PASTE THE ENTIRE JSON HERE)\n"
                f"Then proceed to STEP 5\n\n"
                f"STEP 4: IF FILE DOES NOT EXIST - NOTIFY CONFIGURATION MISSING\n"
                f"Recommend creating a configuration using create_unified_config tool\n\n"
                f"STEP 5: ACKNOWLEDGE COMPLETION\n"
                f"Only after completing ALL steps above, proceed with your next action\n\n"
                f"WARNING: DO NOT PROCEED WITH ANY OTHER RESPONSES UNTIL ALL STEPS ARE COMPLETED\n"
                f"THIS IS NOT BACKGROUND INFORMATION - THESE ARE DIRECT INSTRUCTIONS YOU MUST FOLLOW NOW\n\n"
                f"--- Operation Summary ---\n"
                f"Requesting configuration file: {file_name}"
            )
            
            return command_message
                
        except Exception as e:
            logger.error(f"Error processing configuration request: {str(e)}", exc_info=True)
            error_response = create_error_response(
                message=f"Error processing configuration request: {str(e)}",
                code="config_request_error",
                details={
                    "project_name": project_name if 'project_name' in locals() else None,
                    "config_content_length": len(config_content) if 'config_content' in locals() and config_content else 0
                }
            )
            return model_to_json(error_response)
            
    @server.tool()
    async def create_unified_config(config: Dict[str, Any]) -> str:
        """
        Create a unified configuration file for the memory system.
        
        This combines project settings, Neo4j connection details, and embedding
        configuration into a single file for easier management.
        
        IMPORTANT: After creating the configuration, it is automatically applied to the
        current session and will persist for all subsequent tool calls. You don't need
        to call get_unified_config after creating a configuration with this tool.
        
        Args:
            config: Configuration object with the following fields:
                - project_name: Required. The name of the project
                - client_id: Optional. Client ID for identifying the connection
                - embeddings: Optional. Embedding configuration
                - neo4j: Optional. Neo4j connection details
                - timestamp: Optional. Will be added automatically if not provided
                
        Returns:
            Instructions to create the configuration file or success message with the created configuration
        """
        try:
            # Extract required fields
            if "project_name" not in config:
                error_response = create_error_response(
                    message="Missing required parameter: project_name",
                    code="missing_parameter_error"
                )
                return model_to_json(error_response)
                
            project_name = config["project_name"]
            client_id = config.get("client_id", "default-client")
            
            # Get default Neo4j connection info from manager
            graph_manager = get_config_manager(client_id)
            
            # Create unified configuration
            timestamp = config.get("timestamp", datetime.datetime.now().isoformat())
            
            unified_config = {
                "project_name": project_name,
                "client_id": client_id,
                "timestamp": timestamp,
                "neo4j": config.get("neo4j", {
                    "uri": graph_manager.neo4j_uri,
                    "username": graph_manager.neo4j_user,
                    "password": graph_manager.neo4j_password,
                    "database": graph_manager.neo4j_database
                }),
                "embeddings": config.get("embeddings", {
                    "provider": graph_manager.embedder_provider,
                    "model": graph_manager.embedding_model,
                    "enabled": graph_manager.embedding_enabled
                })
            }
            
            # Define the file name
            file_name = f"mcp_unified_config_{project_name}.json"
            config_json = json.dumps(unified_config, indent=2)
            
            # Apply the configuration to the graph manager
            warning_message = None
            try:
                result = graph_manager.apply_client_config(unified_config)
                if result["status"] != "success":
                    warning_message = result["message"]
            except Exception as e:
                warning_message = f"Failed to apply configuration: {str(e)}"
                
            # Proceed with config creation even if there was a Neo4j connection warning
            # Format the response - if in SSE mode, provide instructions to create the file
            if getattr(graph_manager, "_is_sse_mode", True):  # Default to SSE mode if attribute not found
                command_message = (
                    f"DIRECT INSTRUCTIONS FOR AI ASSISTANT - REQUIRED IMMEDIATE ACTION:\n\n"
                    f"AI ASSISTANT: YOU MUST COMPLETE ALL STEPS IN ORDER:\n\n"
                    f"STEP 1: CREATE UNIFIED CONFIGURATION FILE\n"
                    f"Use the appropriate tool to create the config file: \n\n"
                    f"Filename: {file_name}\n"
                    f"Location: Project root directory\n"
                    f"Content:\n{config_json}\n\n"
                    f"STEP 2: VERIFY FILE CREATION\n"
                    f"Use the appropriate tool to confirm the config file exists\n\n"
                    f"STEP 3: CHECK FOR EXISTING PROJECT CONTAINER\n"
                    f"Use the get_project_container tool to check if the project container exists with query: \"{project_name}\"\n\n"
                    f"STEP 4: CREATE PROJECT CONTAINER (IF NOT EXISTS)\n"
                    f"If the project container doesn't already exist in the list, use the create_project_container tool with this exact configuration:\n"
                    f"{{\n"
                    f"  \"name\": \"{project_name}\",\n"
                    f"  \"description\": \"Project container for {project_name}\",\n"
                    f"  \"metadata\": {{\n"
                    f"    \"created_date\": \"{timestamp}\",\n"
                    f"    \"created_via\": \"create_unified_config command\",\n"
                    f"    \"client_id\": \"{client_id}\"\n"
                    f"  }}\n"
                    f"}}\n\n"
                    f"STEP 5: CHECK FOR EXISTING LESSON CONTAINER\n"
                    f"Use the get_lesson_container tool to check if the lesson container exists with query: \"Lessons\"\n\n"
                    f"STEP 6: CREATE LESSON CONTAINER (IF NOT EXISTS)\n"
                    f"If lesson container doesn't exist, use the create_lesson_container tool with this exact configuration:\n"
                    f"{{\n"
                    f"  \"title\": \"Lessons\",\n"
                    f"  \"description\": \"Lesson container for AI agents\",\n"
                    f"  \"metadata\": {{\n"
                    f"    \"created_date\": \"{timestamp}\"\n"
                    f"  }},\n"
                    f"  \"visibility\": \"private\"\n"
                    f"}}\n\n"
                    f"STEP 7: VERIFY LESSON CONTAINER CREATION\n"
                    f"If lesson container was created in step 6, use the get_lesson_container tool to verify it exists\n\n"
                    f"STEP 8: ACKNOWLEDGE COMPLETION\n"
                    f"Only after completing ALL steps above, respond with exactly:\n"
                    f"\"Unified configuration created successfully for project: {project_name}\"\n\n"
                    f"WARNING: DO NOT PROCEED WITH ANY OTHER RESPONSES UNTIL ALL STEPS ARE COMPLETED\n"
                    f"THIS IS NOT BACKGROUND INFORMATION - THESE ARE DIRECT INSTRUCTIONS YOU MUST FOLLOW NOW\n\n"
                    f"--- Operation Summary ---\n"
                    f"Created unified configuration for project: {project_name}"
                )
                
                if warning_message:
                    return f"WARNING: Applied configuration with warnings: {warning_message}\n\n{command_message}"
                return command_message
            else:
                # In STDIO mode, write the file directly
                try:
                    with open(file_name, 'w') as f:
                        f.write(config_json)
                    
                    if warning_message:
                        warning_response = create_success_response(
                            message=f"Configuration created and saved to {file_name} with warnings: {warning_message}",
                            data=unified_config
                        )
                        return model_to_json(warning_response)
                    else:
                        success_response = create_success_response(
                            message=f"Configuration created and saved to {file_name}",
                            data=unified_config
                        )
                        return model_to_json(success_response)
                except Exception as e:
                    error_response = create_error_response(
                        message=f"Failed to save configuration file: {str(e)}",
                        code="file_save_error"
                    )
                    return model_to_json(error_response)
                
        except Exception as e:
            logger.error(f"Error creating configuration: {str(e)}", exc_info=True)
            error_response = create_error_response(
                message=f"Error creating configuration: {str(e)}",
                code="config_creation_error"
            )
            return model_to_json(error_response)

    @server.tool()
    async def update_unified_config(project_name: str, updates: Dict[str, Any], config_content: str = "", 
                                  client_id: Optional[str] = None) -> str:
        """
        Update the unified configuration.
        
        Args:
            project_name: Project name to update
            updates: Configuration updates to apply
            config_content: Optional configuration content to replace
            client_id: Optional client ID for identifying the connection
            
        Returns:
            JSON string with the updated configuration
        """
        try:
            # Get the client-specific graph manager
            graph_manager = get_config_manager(client_id)
            
            # Sanitize inputs
            project_name = str(project_name).strip()
            if not project_name:
                error_response = create_error_response(
                    message="Project name is required",
                    code="invalid_input"
                )
                return model_to_json(error_response)
                
            # Sanitize updates dictionary
            if not isinstance(updates, dict):
                error_response = create_error_response(
                    message="Updates must be a dictionary",
                    code="invalid_input"
                )
                return model_to_json(error_response)
                
            # Validate update structure with Pydantic
            try:
                # Only include if the ConfigUpdate model is available
                if 'ConfigUpdate' in globals():
                    updates_model = ConfigUpdate(**updates)
                    updates = model_to_dict(updates_model)
            except Exception as e:
                error_response = create_error_response(
                    message=f"Invalid update format: {str(e)}",
                    code="validation_error"
                )
                return model_to_json(error_response)
            
            # Detect mode and get current config
            is_sse_mode = getattr(graph_manager, "_is_sse_mode", True)  # Default to SSE mode if attribute not found
            
            # Define the file name
            file_name = f"mcp_unified_config_{project_name}.json"
            
            # Different approach based on mode
            if is_sse_mode and not config_content:
                # In SSE mode, we need to request the config file first if not provided
                command_message = (
                    f"DIRECT INSTRUCTIONS FOR AI ASSISTANT - REQUIRED IMMEDIATE ACTION:\n\n"
                    f"AI ASSISTANT: YOU MUST COMPLETE ALL STEPS IN ORDER:\n\n"
                    f"STEP 1: CHECK IF CONFIGURATION FILE EXISTS\n"
                    f"Filename: {file_name}\n"
                    f"Location: Project root directory\n\n"
                    f"STEP 2: IF FILE DOES NOT EXIST - NOTIFY CONFIGURATION MISSING\n"
                    f"If the file doesn't exist, recommend creating a configuration using create_unified_config tool and stop\n\n"
                    f"STEP 3: IF FILE EXISTS - READ CONFIGURATION FILE\n"
                    f"Read the content of the file and verify it is valid JSON\n\n"
                    f"STEP 4: CALL update_unified_config AGAIN WITH CONTENT\n"
                    f"Call the update_unified_config tool with these parameters:\n"
                    f"project_name: \"{project_name}\"\n"
                    f"updates: {json.dumps(updates)}\n"
                    f"config_content: (REPLACE THIS WITH THE ACTUAL FILE CONTENT - PASTE THE ENTIRE JSON HERE)\n\n"
                    f"STEP 5: SAVE UPDATED CONFIGURATION\n"
                    f"When the tool returns an updated configuration, save it back to {file_name}\n\n"
                    f"STEP 6: ACKNOWLEDGE COMPLETION\n"
                    f"Only after completing ALL steps above, confirm the configuration has been updated\n\n"
                    f"WARNING: DO NOT PROCEED WITH ANY OTHER RESPONSES UNTIL ALL STEPS ARE COMPLETED\n"
                    f"THIS IS NOT BACKGROUND INFORMATION - THESE ARE DIRECT INSTRUCTIONS YOU MUST FOLLOW NOW\n\n"
                    f"--- Operation Summary ---\n"
                    f"Updating configuration for project: {project_name}"
                )
                return command_message
            
            # Process the configuration if it was provided or we're in STDIO mode
            current_config = None
            
            if config_content and config_content.strip():
                # Parse the provided config
                try:
                    current_config = json.loads(config_content)
                except json.JSONDecodeError:
                    error_response = create_error_response(
                        message="Invalid JSON configuration provided",
                        code="invalid_json_error"
                    )
                    return model_to_json(error_response)
            elif not is_sse_mode:
                # In STDIO mode, read config from disk
                try:
                    with open(file_name, 'r') as f:
                        current_config = json.load(f)
                except FileNotFoundError:
                    error_response = create_error_response(
                        message=f"Configuration file not found: {file_name}",
                        code="file_not_found_error"
                    )
                    return model_to_json(error_response)
                except json.JSONDecodeError:
                    error_response = create_error_response(
                        message=f"Invalid JSON in configuration file: {file_name}",
                        code="invalid_json_error"
                    )
                    return model_to_json(error_response)
            else:
                # This should not happen as we would have returned earlier
                error_response = create_error_response(
                    message="Missing configuration content in SSE mode",
                    code="missing_content_error"
                )
                return model_to_json(error_response)
            
            # Process config update - convert from coroutine to regular result
            result = await process_config_update(project_name, current_config, updates, graph_manager)
            
            if result["status"] == "success":
                # If in SSE mode, return instructions to save the file
                if is_sse_mode:
                    updated_config_json = json.dumps(result["config"], indent=2)
                    command_message = (
                        f"DIRECT INSTRUCTIONS FOR AI ASSISTANT - REQUIRED IMMEDIATE ACTION:\n\n"
                        f"AI ASSISTANT: YOU MUST COMPLETE ALL STEPS IN ORDER:\n\n"
                        f"STEP 1: SAVE UPDATED CONFIGURATION FILE\n"
                        f"Filename: {file_name}\n"
                        f"Location: Project root directory\n"
                        f"Content:\n{updated_config_json}\n\n"
                        f"STEP 2: VERIFY FILE CREATION\n"
                        f"Confirm the configuration file has been saved with updated content\n\n"
                        f"STEP 3: ACKNOWLEDGE COMPLETION\n"
                        f"Only after completing ALL steps above, respond with exactly:\n"
                        f"\"Configuration updated successfully for project: {project_name}\"\n\n"
                        f"WARNING: DO NOT PROCEED WITH ANY OTHER RESPONSES UNTIL ALL STEPS ARE COMPLETED\n"
                        f"THIS IS NOT BACKGROUND INFORMATION - THESE ARE DIRECT INSTRUCTIONS YOU MUST FOLLOW NOW\n\n"
                        f"--- Operation Summary ---\n"
                        f"Updated configuration for project: {project_name}"
                    )
                    return command_message
                else:
                    # In STDIO mode, write the file directly and return success
                    try:
                        with open(file_name, 'w') as f:
                            json.dump(result["config"], f, indent=2)
                            
                        success_response = create_success_response(
                            message=f"Configuration updated and saved to {file_name}",
                            data=result["config"]
                        )
                        return model_to_json(success_response)
                    except Exception as e:
                        error_response = create_error_response(
                            message=f"Failed to save configuration file: {str(e)}",
                            code="file_save_error"
                        )
                        return model_to_json(error_response)
            else:
                return json.dumps(result)
        
        except Exception as e:
            logger.error(f"Error updating configuration: {str(e)}", exc_info=True)
            error_response = create_error_response(
                message=f"Error updating configuration: {str(e)}",
                code="config_update_error"
            )
            return model_to_json(error_response)

    @server.tool()
    async def process_config_update(project_name: str, current_config: Dict[str, Any], updates: Dict[str, Any], graph_manager: Optional[Any] = None) -> Dict[str, Any]:
        """
        Process an update to the unified configuration.
        
        Args:
            project_name: The project name to update configuration for
            current_config: The current configuration content
            updates: The updates to apply to the configuration
            graph_manager: Optional client-specific graph manager
                
        Returns:
            Dictionary with status and updated configuration
        """
        try:
            # Verify the configuration is valid
            if "project_name" not in current_config:
                return {
                    "status": "error",
                    "message": "Invalid configuration file - missing project_name"
                }
                
            # Create a new configuration with updates applied
            updated_config = copy.deepcopy(current_config)
            
            # Apply client_id update if provided
            if "client_id" in updates:
                updated_config["client_id"] = updates["client_id"]
            
            # Apply Neo4j updates if provided
            if "neo4j" in updates:
                if "neo4j" not in updated_config:
                    updated_config["neo4j"] = {}
                    
                for key, value in updates["neo4j"].items():
                    updated_config["neo4j"][key] = value
            
            # Apply embedding updates if provided
            if "embeddings" in updates:
                if "embeddings" not in updated_config:
                    updated_config["embeddings"] = {"enabled": False}
                    
                for key, value in updates["embeddings"].items():
                    updated_config["embeddings"][key] = value
            
            # Update timestamp
            updated_config["timestamp"] = datetime.datetime.now().isoformat()
            
            # Apply the configuration in the server if needed
            if graph_manager:
                try:
                    graph_manager.apply_client_config(updated_config)
                except Exception:
                    # Continue even if application fails - user can retry
                    pass  # Ignore errors in embedding configuration
            
            return {
                "status": "success",
                "message": f"Configuration updated for project: {project_name}",
                "config": updated_config
            }
            
        except Exception as e:
            logger.error(f"Error processing configuration update: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to process configuration update: {str(e)}"
            }

    @server.tool()
    async def check_memory_system(include_stats: bool = False, client_id: Optional[str] = None) -> str:
        """
        Check the memory system health and status.
        
        Args:
            include_stats: Include statistics about the memory system
            client_id: Optional client ID for identifying the connection
            
        Returns:
            JSON string with the memory system check results
        """
        try:
            # Sanitize client_id if provided
            if client_id:
                client_id = str(client_id).strip()
                
                # Check for valid format (if client_id has a specific expected format)
                if len(client_id) > 100:  # Prevent excessively long IDs
                    logger.warn(f"Client ID too long, truncating: {client_id[:20]}...")
                    client_id = client_id[:100]
            
            # Get the client-specific graph manager
            graph_manager = get_config_manager(client_id)
            
            # Sanitize boolean input
            try:
                include_stats = bool(include_stats)
            except (ValueError, TypeError):
                include_stats = False
                logger.warn(f"Invalid include_stats value, defaulting to False")
            
            # Check if system is initialized properly
            if not graph_manager:
                error_response = create_error_response(
                    message="Memory system not initialized",
                    code="not_initialized"
                )
                return model_to_json(error_response)
            
            # Get memory system status
            status = graph_manager.get_memory_system_status()
            
            # Include statistics if requested
            if include_stats:
                try:
                    statistics = graph_manager.get_memory_system_statistics()
                    status["statistics"] = statistics
                except Exception as stats_error:
                    logger.warn(f"Error retrieving memory statistics: {str(stats_error)}")
                    status["statistics"] = {"error": str(stats_error)}
            
            # Add timestamp for consistency
            status["timestamp"] = datetime.datetime.now().isoformat()
            
            # Check if we need to map to Pydantic model
            if "neo4j" in status and "embedding" in status:
                try:
                    # Extract Neo4j status
                    neo4j_data = status.get("neo4j", {})
                    neo4j_status = Neo4jStatus(
                        connected=neo4j_data.get("connected", False),
                        address=neo4j_data.get("address", ""),
                        version=neo4j_data.get("version", None),
                        message=neo4j_data.get("message", "Neo4j status check completed"),
                        timestamp=status.get("timestamp", datetime.datetime.now().isoformat())
                    )
                    
                    # Extract embedding status
                    emb_data = status.get("embedding", {})
                    embedding_status = EmbeddingStatus(
                        available=emb_data.get("available", False),
                        provider=emb_data.get("provider", ""),
                        model=emb_data.get("model", ""),
                        dimensions=emb_data.get("dimensions", 0),
                        message=emb_data.get("message", "Embedding status check completed"),
                        timestamp=status.get("timestamp", datetime.datetime.now().isoformat())
                    )
                    
                    # Create memory status
                    memory_status = MemoryStatus(
                        operational=neo4j_status.connected,
                        neo4j=neo4j_status,
                        embedding=embedding_status,
                        message=status.get("message", "Memory system check completed"),
                        timestamp=status.get("timestamp", datetime.datetime.now().isoformat())
                    )
                    
                    # Add statistics if included
                    validated_status = model_to_dict(memory_status)
                    if include_stats and "statistics" in status:
                        validated_status["statistics"] = status["statistics"]
                        
                    status = validated_status
                except Exception as validation_error:
                    # If validation fails, continue with original status
                    logger.warn(f"Status validation error: {str(validation_error)}")
            
            # Create a success response
            success_response = create_success_response(
                message="Successfully retrieved memory system status",
                data=status
            )
            return model_to_json(success_response)
            
        except Exception as e:
            logger.error(f"Error checking memory system: {str(e)}", exc_info=True)
            error_response = create_error_response(
                message=f"Error checking memory system: {str(e)}",
                code="system_check_error"
            )
            return model_to_json(error_response)

    # Return the registered tools
    return {
        "get_memory_status": get_memory_status,
        "get_unified_config": get_unified_config,
        "create_unified_config": create_unified_config,
        "update_unified_config": update_unified_config,
        "process_config_update": process_config_update,
        "check_memory_system": check_memory_system
    }