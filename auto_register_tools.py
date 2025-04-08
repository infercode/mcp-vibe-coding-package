#!/usr/bin/env python3
"""
Tool Registration Helper

This script automatically adds @register_function decorators to all functions
that are decorated with @server.tool() in the codebase.
"""

import os
import re
import sys
from pathlib import Path

# Regex pattern to find functions with @server.tool() decorator
SERVER_TOOL_PATTERN = r'@server\.tool\(\)([\s\n]+)async def ([a-zA-Z0-9_]+)\('

# Target directories to search for files
TARGET_DIRS = [
    'src/registry',
    'src/tools'
]

# Files to skip (already manually updated)
SKIP_FILES = [
    'src/registry/health_diagnostics.py',
    'src/registry/feedback_mechanism.py',
]

def add_register_function_decorator(file_path):
    """
    Add @register_function decorator to all @server.tool() functions in a file.
    
    Args:
        file_path: Path to the file to modify
        
    Returns:
        Tuple of (success, message, modified_count)
    """
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check if we already imported register_function
        if 'from src.registry.registry_manager import register_function' not in content:
            # Add import for register_function if not present
            if 'from src.registry.registry_manager import' in content:
                # Modify existing import to add register_function
                content = re.sub(
                    r'from src\.registry\.registry_manager import ([^,\n]+)', 
                    r'from src.registry.registry_manager import \1, register_function', 
                    content
                )
            else:
                # Add new import for register_function
                import_line = 'from src.registry.registry_manager import register_function\n'
                # Insert after other imports
                imports_end = re.search(r'(^import.*?\n+|^from.*?\n+)+', content, re.MULTILINE)
                if imports_end:
                    pos = imports_end.end()
                    content = content[:pos] + import_line + content[pos:]
                else:
                    # Insert at top if no other imports
                    content = import_line + content
                    
        # Find all @server.tool() decorated functions
        matches = re.finditer(SERVER_TOOL_PATTERN, content, re.MULTILINE)
        modified_count = 0
        
        # Collect all replacements to make to avoid overlapping matches
        replacements = []
        for match in matches:
            whitespace = match.group(1)
            func_name = match.group(2)
            
            # Determine namespace from context
            namespace = 'registry'  # Default namespace
            
            # Try to determine namespace from the file path or other hints
            if 'feedback_mechanism.py' in file_path:
                namespace = 'feedback'
            elif 'health_diagnostics.py' in file_path:
                namespace = 'health'
            elif 'core_memory_tools.py' in file_path:
                namespace = 'memory'
            elif 'project_memory_tools.py' in file_path:
                namespace = 'project'
            elif 'lesson_memory_tools.py' in file_path:
                namespace = 'lesson'
            elif 'config_tools.py' in file_path:
                namespace = 'config'
            
            # Create decorator to insert
            decorator = f'@register_function("{namespace}", "{func_name}")'
            
            # Add replacement to list
            replacement = (match.start(), f'{decorator}{whitespace}@server.tool()')
            replacements.append(replacement)
            modified_count += 1
            
        # Apply replacements in reverse order to maintain correct positions
        replacements.sort(reverse=True)
        for pos, text in replacements:
            content = content[:pos] + text + content[pos:]
            
        # Write updated content back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return True, f"Modified {modified_count} functions", modified_count
    
    except Exception as e:
        return False, f"Error: {str(e)}", 0

def process_files():
    """Process all Python files in target directories."""
    print(f"Starting tool registration helper...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    modified_files = 0
    modified_functions = 0
    
    for dir_name in TARGET_DIRS:
        dir_path = os.path.join(base_dir, dir_name)
        print(f"Scanning directory: {dir_path}")
        
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, base_dir)
                    
                    # Skip already updated files
                    if rel_path in SKIP_FILES:
                        print(f"Skipping {rel_path} (already updated)")
                        continue
                    
                    print(f"Processing {rel_path}...")
                    success, message, count = add_register_function_decorator(file_path)
                    
                    if success and count > 0:
                        modified_files += 1
                        modified_functions += count
                        print(f"  ✓ {message}")
                    elif success:
                        print(f"  ✓ No modifications needed")
                    else:
                        print(f"  ✗ {message}")
    
    print("\nSummary:")
    print(f"- Modified {modified_functions} functions in {modified_files} files")
    print(f"- Target directories: {', '.join(TARGET_DIRS)}")
    print("- Skipped files: {', '.join(SKIP_FILES)}")
    
if __name__ == "__main__":
    process_files() 