"""
Memory persistence utilities for ACDEK.
This module provides functionality for storing and retrieving user memory traces.
"""

from typing import Dict, List, Any, Optional
import json
import os
import time
from datetime import datetime

# In-memory storage for mock DB (would be replaced with actual DB in production)
_memory_store = {}

def save_memory_trace(user_id: str, memory_trace: Dict[str, Any]) -> bool:
    """
    Save a user's memory trace to the store.
    
    Args:
        user_id: The user identifier
        memory_trace: The memory trace data to store
        
    Returns:
        True if successful, False otherwise
    """
    if not user_id or not memory_trace:
        return False
        
    # Add timestamp for this update
    memory_trace["_last_updated"] = datetime.now().isoformat()
    
    # Store in memory
    _memory_store[user_id] = memory_trace
    
    # For demo purposes, we could write to a JSON file too
    try:
        # Create directory if it doesn't exist
        os.makedirs('mock_db/memory_cache', exist_ok=True)
        
        # Write to file named after user_id
        with open(f'mock_db/memory_cache/{user_id}.json', 'w') as f:
            json.dump(memory_trace, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving memory trace: {str(e)}")
        return False

def get_memory_trace(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a user's memory trace from the store.
    
    Args:
        user_id: The user identifier
        
    Returns:
        The memory trace as a dictionary, or None if not found
    """
    # First check in-memory store
    if user_id in _memory_store:
        return _memory_store[user_id]
        
    # Then try to load from file
    try:
        memory_file = f'mock_db/memory_cache/{user_id}.json'
        if os.path.exists(memory_file):
            with open(memory_file, 'r') as f:
                memory_trace = json.load(f)
                # Cache it in memory for future use
                _memory_store[user_id] = memory_trace
                return memory_trace
    except Exception as e:
        print(f"Error loading memory trace: {str(e)}")
    
    return None

def update_session_history(user_id: str, 
                           session_data: Dict[str, Any]) -> bool:
    """
    Add a new session to the user's session history.
    
    Args:
        user_id: The user identifier
        session_data: Data about the current session to add
        
    Returns:
        True if successful, False otherwise
    """
    memory_trace = get_memory_trace(user_id)
    
    if not memory_trace:
        # Create new memory trace with session history
        memory_trace = {
            "session_history": [session_data],
            "preference_history": {},
            "learned_traits": {}
        }
    else:
        # Add to existing session history
        if "session_history" not in memory_trace:
            memory_trace["session_history"] = []
        
        memory_trace["session_history"].append(session_data)
    
    return save_memory_trace(user_id, memory_trace)

def update_learned_traits(user_id: str, 
                          trait_updates: Dict[str, float],
                          learning_rate: float = 0.2) -> bool:
    """
    Update the learned traits for a user, with exponential moving average.
    
    Args:
        user_id: The user identifier
        trait_updates: New trait values to incorporate
        learning_rate: How quickly to adapt to new information (0-1)
        
    Returns:
        True if successful, False otherwise
    """
    memory_trace = get_memory_trace(user_id)
    
    if not memory_trace:
        # Create new memory trace with these traits
        memory_trace = {
            "session_history": [],
            "preference_history": {},
            "learned_traits": trait_updates
        }
    else:
        # Update existing traits with exponential moving average
        if "learned_traits" not in memory_trace:
            memory_trace["learned_traits"] = {}
            
        for trait, value in trait_updates.items():
            if trait in memory_trace["learned_traits"]:
                # Apply exponential moving average
                current = memory_trace["learned_traits"][trait]
                memory_trace["learned_traits"][trait] = (
                    (1 - learning_rate) * current + learning_rate * value
                )
            else:
                # First time seeing this trait
                memory_trace["learned_traits"][trait] = value
    
    return save_memory_trace(user_id, memory_trace)