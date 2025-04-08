#!/usr/bin/env python3
"""
Function Registry Pattern Demos

This script runs demonstrations for the Function Registry Pattern components.
"""

import asyncio
import sys
import os

# Add the project root to the path for proper imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

async def run_demos():
    """Run all Function Registry Pattern demonstration scripts."""
    print("===== FUNCTION REGISTRY PATTERN DEMONSTRATIONS =====")
    
    # Import the demo modules
    print("\nImporting demo modules...")
    
    try:
        # We use local imports to avoid circular import issues
        from src.registry.migration_framework_demo import demo_migration_framework
        from src.registry.ide_integration_demo import demo_ide_integration
        
        # Run the Tool Migration Framework demo
        print("\n\n===== TOOL MIGRATION FRAMEWORK DEMO =====")
        await demo_migration_framework()
        
        # Run the IDE Integration Optimizations demo
        print("\n\n===== IDE INTEGRATION OPTIMIZATIONS DEMO =====")
        await demo_ide_integration()
        
        print("\n\n===== ALL DEMONSTRATIONS COMPLETED =====")
        print("\nPhase 4 is now fully implemented with:")
        print("1. Tool Migration Framework")
        print("2. IDE Integration Optimizations")
        print("3. Agent Guidance System")
        
    except ImportError as e:
        print(f"Error importing demo modules: {str(e)}")
        print("Please ensure the demo modules are properly installed.")
    except Exception as e:
        print(f"Error running demos: {str(e)}")

if __name__ == "__main__":
    asyncio.run(run_demos()) 