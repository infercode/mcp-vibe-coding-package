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
            # Get the client-specific graph manager
            graph_manager = get_config_manager(client_id)
            
            # Get basic information about the memory system
            config = graph_manager.get_current_config()
            
            # Get Neo4j connection info
            neo4j_connected = graph_manager.check_connection()
            
            # Build status object
            status = {
                "neo4j": {
                    "connected": neo4j_connected,
                    "uri": graph_manager.neo4j_uri,
                    "database": graph_manager.neo4j_database
                },
                "embeddings": {
                    "enabled": graph_manager.embedding_enabled,
                    "provider": graph_manager.embedder_provider,
                    "model": graph_manager.embedding_model
                },
                "project": graph_manager.default_project_name,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Create a success response
            success_response = create_success_response(
                message="Successfully retrieved memory status",
                data=status
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
    async def get_unified_config(project_name: str = "", config_content: str = "") -> str:
        """
        Retrieve and apply the unified configuration file from the client's project.
        
        IMPORTANT: This tool only needs to be called ONCE PER SESSION. The configuration 
        will persist for all subsequent tool calls in the same session. You only need to
        call it again when:
        1. Starting a new session
        2. Changing to a different project
        3. Needing to update configuration settings
        
        This tool serves two purposes:
        1. When called without config_content: It requests the client to read the configuration file
        2. When called with config_content: It applies the received configuration to the memory manager
        
        Args:
            project_name: Optional project name to determine the config file name.
                          If not provided, uses the default config file.
            config_content: Optional JSON string with the configuration content from the client.
                          When provided, applies the configuration to the memory system.
                
        Returns:
            JSON string with the configuration file content or instructions to create one
        """
        try:
            # Define the default config file name
            file_name = "mcp_unified_config.json"
            if project_name and project_name.strip():
                file_name = f"mcp_unified_config_{project_name}.json"
            
            # Check if config content was provided
            if config_content and config_content.strip():
                # Parse the config content
                try:
                    config = json.loads(config_content)
                    
                    # Extract client_id from config if present
                    client_id = config.get("client_id", None)
                    
                    # Get the client-specific graph manager
                    graph_manager = get_config_manager(client_id)
                    
                    # Apply the configuration
                    result = graph_manager.apply_client_config(config)
                    
                    # Return success or error message
                    if result["status"] == "success":
                        success_response = create_success_response(
                            message="Configuration applied successfully",
                            data=graph_manager.get_current_config()
                        )
                        return model_to_json(success_response)
                    else:
                        error_response = create_error_response(
                            message=result["message"],
                            code="config_application_error"
                        )
                        return model_to_json(error_response)
                        
                except json.JSONDecodeError:
                    error_response = create_error_response(
                        message="Invalid JSON configuration provided",
                        code="invalid_json_error"
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
                code="config_request_error"
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
    async def update_unified_config(project_name: str, updates: Dict[str, Any], config_content: str = "") -> str:
        """
        Update specific parts of the unified configuration.
        
        This tool allows you to update specific parts of the configuration without
        recreating the entire file. The updates will be applied to both the configuration
        file and the active memory system for the current session.
        
        NOTE: After updating, the changes will persist for all subsequent tool calls in the
        same session without needing to call get_unified_config again.
        
        Args:
            project_name: The project name to determine the config file name
            updates: Dictionary with updates to apply to the configuration
            config_content: Optional JSON string with the current configuration content from the client (for SSE mode).
                When provided, applies updates to this configuration; when not provided, reads from server (STDIO mode).
        
        Returns:
            JSON string with the updated configuration or instructions to update it
        """
        try:
            # Get the client-specific graph manager
            graph_manager = get_config_manager(None)  # We'll extract client_id from config
            
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

    # Return the registered tools
    return {
        "get_memory_status": get_memory_status,
        "get_unified_config": get_unified_config,
        "create_unified_config": create_unified_config,
        "update_unified_config": update_unified_config,
        "process_config_update": process_config_update
    }