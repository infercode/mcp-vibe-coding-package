#!/usr/bin/env python3
"""
Hybrid MCP Server - Entry Point

This script launches a Hybrid MCP Server that registers all functions
with the internal registry but only exposes essential tools to clients.
"""

import sys
import os

if __name__ == "__main__":
    # Add repository root to path if needed
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    
    # Import and run the main function from the hybrid server module
    try:
        from src.hybrid_mcp_server import main
        sys.exit(main())
    except ImportError as e:
        print(f"Error importing hybrid_mcp_server module: {e}")
        sys.exit(1) 