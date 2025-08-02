"""Validation utilities for ensuring config dictionaries and object states are correct."""

from typing import Dict, Any, List, Optional, Union


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_agent_mu_config(config: Dict[str, Any]) -> None:
    """
    Validate agent_mu_config dictionary structure.
    
    Args:
        config: Dictionary containing agent mu configuration
        
    Raises:
        ValidationError: If config is invalid
    """
    if not isinstance(config, dict):
        raise ValidationError("agent_mu_config must be a dictionary")
    
    required_keys = ['default_mu', 'default_sigma']
    for key in required_keys:
        if key not in config:
            config[key] = 50.0 if key == 'default_mu' else 10.0



def ensure_dictionary_keys(d: Dict[str, Any], required_keys: List[str], defaults: Dict[str, Any] = None) -> None:
    """
    Ensure a dictionary has all required keys, adding defaults if needed.
    
    Args:
        d: Dictionary to validate and update
        required_keys: List of keys that must exist
        defaults: Dictionary of default values for missing keys
        
    Raises:
        ValidationError: If dictionary is invalid and no defaults provided
    """
    if defaults is None:
        defaults = {}
    
    missing_keys = []
    for key in required_keys:
        if key not in d:
            if key in defaults:
                d[key] = defaults[key]
            else:
                missing_keys.append(key)
    
    if missing_keys:
        raise ValidationError(f"Dictionary missing required keys: {missing_keys}")