"""
Validation utilities for ensuring config dictionaries and object states are correct.
This module helps replace hasattr() and .get() patterns with explicit validation.
"""

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


def validate_dcop_config(config: Dict[str, Any]) -> None:
    """
    Validate DCOP configuration dictionary.
    
    Args:
        config: Dictionary containing DCOP configuration
        
    Raises:
        ValidationError: If config is invalid
    """
    if not isinstance(config, dict):
        raise ValidationError("DCOP config must be a dictionary")
    
    # Ensure required keys exist with defaults
    defaults = {
        'p0': 0.5,
        'learning_rate': 0.01,
        'baseline_decay': 0.9,
        'iteration_per_episode': 20,
        'num_episodes': 1
    }
    
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value


def ensure_agent_has_required_methods(agent: Any, required_methods: List[str]) -> None:
    """
    Ensure an agent has all required methods.
    
    Args:
        agent: Agent object to validate
        required_methods: List of method names that must exist
        
    Raises:
        ValidationError: If any required method is missing
    """
    missing_methods = []
    for method_name in required_methods:
        if not hasattr(agent, method_name):
            missing_methods.append(method_name)
    
    if missing_methods:
        raise ValidationError(
            f"Agent {type(agent).__name__} missing required methods: {missing_methods}"
        )


def ensure_agent_has_required_attributes(agent: Any, required_attributes: List[str]) -> None:
    """
    Ensure an agent has all required attributes.
    
    Args:
        agent: Agent object to validate
        required_attributes: List of attribute names that must exist
        
    Raises:
        ValidationError: If any required attribute is missing
    """
    missing_attributes = []
    for attr_name in required_attributes:
        if not hasattr(agent, attr_name):
            missing_attributes.append(attr_name)
    
    if missing_attributes:
        raise ValidationError(
            f"Agent {type(agent).__name__} missing required attributes: {missing_attributes}"
        )


def validate_learning_statistics_config(config: Dict[str, Any]) -> None:
    """
    Validate learning statistics configuration.
    
    Args:
        config: Dictionary containing learning statistics config
        
    Raises:
        ValidationError: If config is invalid
    """
    if not isinstance(config, dict):
        raise ValidationError("Learning statistics config must be a dictionary")
    
    # Set defaults for any missing keys
    defaults = {
        'agent_mu_values': {},
        'all_episode_costs': [],
        'agent_policies': {},
        'performance': {}
    }
    
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value


def validate_feature_stats(stats: Any) -> Dict[str, Any]:
    """
    Validate and normalize feature statistics.
    
    Args:
        stats: Feature statistics object or dictionary
        
    Returns:
        Normalized dictionary representation
    """
    if stats is None:
        return {}
    
    if hasattr(stats, 'copy') and callable(stats.copy):
        return stats.copy()
    
    if isinstance(stats, dict):
        return stats.copy()
    
    return {}


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