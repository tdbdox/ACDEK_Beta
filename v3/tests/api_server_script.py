#!/usr/bin/env python3
"""
Start the ACDEK API server
"""
import uvicorn
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def main():
    # Get the full path to the main.py file
    api_dir = Path(__file__).resolve().parent.parent / "web" / "api"
    main_path = api_dir / "main.py"
    
    # Start the server with the full module path
    module_path = "web.api.main:app"
    print(f"Starting server with module path: {module_path}")
    uvicorn.run(module_path, host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()