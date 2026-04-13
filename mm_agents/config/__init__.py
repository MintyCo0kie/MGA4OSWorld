"""
Configuration module for MGA Agent.

Usage:
    from mm_agents.config import load_config, MGA_Config
    
    # Load from YAML file
    config = load_config("path/to/config.yaml")
    
    # Access config
    print(config.planner.model)
    print(config.common.api_key)
"""

from .config_loader import load_config, MGA_Config

__all__ = ["load_config", "MGA_Config"]