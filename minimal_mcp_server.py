#!/usr/bin/env python3
"""
Minimal MCP Server - Entry Point

This script launches a Minimal MCP Server that only registers and exposes
the three essential registry tools, while allowing access to all functions.
"""

import sys
import os

if __name__ == "__main__":
    # Add repository root to path if needed
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    
    # Import and run the main function from the minimal server module
    try:
        from src.minimal_mcp_server import main
        sys.exit(main())
    except ImportError as e:
        print(f"Error importing minimal_mcp_server module: {e}")
        sys.exit(1) 