"""
Solver configuration for PDPTWT.

This module defines configuration dataclasses for solver parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from loguru import logger


@dataclass
class SolverConfig:
    """Configuration for optimization solver."""
    
    # Solver selection
    name: str = "gurobi"
    
    # Time limits
    time_limit: int = 3600  # seconds
    
    # Optimality tolerances
    mip_gap: float = 0.01  # 1%
    
    # Computational resources
    threads: int = 0  # 0 = auto-detect
    
    # Solver-specific parameters
    symmetry: int = 0  # 0 = auto, -1 = off, 1 = conservative, 2 = aggressive
    
    # Logging
    log_to_console: bool = True
    log_file: Optional[Path] = None
    
    # Additional parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, filepath: Path | str) -> 'SolverConfig':
        """
        Load configuration from YAML file.
        
        Args:
            filepath: Path to YAML configuration file
            
        Returns:
            SolverConfig instance
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract solver section
        solver_config = config_dict.get('solver', {})
        
        return cls(
            name=solver_config.get('name', 'gurobi'),
            time_limit=solver_config.get('time_limit', 3600),
            mip_gap=solver_config.get('mip_gap', 0.01),
            threads=solver_config.get('threads', 0),
            symmetry=solver_config.get('symmetry', 0),
            log_to_console=solver_config.get('log_to_console', True),
            log_file=Path(solver_config['log_file']) if solver_config.get('log_file') else None,
            extra_params=solver_config.get('extra_params', {})
        )
    
    @classmethod
    def from_profile(cls, profile_name: str, profiles_file: Path | str) -> 'SolverConfig':
        """
        Load configuration from a named profile.
        
        Args:
            profile_name: Name of the profile (e.g., 'quick', 'standard', 'thorough')
            profiles_file: Path to profiles YAML file
            
        Returns:
            SolverConfig instance
        """
        profiles_file = Path(profiles_file)
        
        if not profiles_file.exists():
            raise FileNotFoundError(f"Profiles file not found: {profiles_file}")
        
        with open(profiles_file, 'r') as f:
            profiles = yaml.safe_load(f)
        
        if profile_name not in profiles:
            available = ', '.join(profiles.keys())
            raise ValueError(f"Profile '{profile_name}' not found. Available: {available}")
        
        profile = profiles[profile_name]
        logger.info(f"Loading solver profile '{profile_name}': {profile.get('description', '')}")
        
        return cls(
            time_limit=profile.get('time_limit', 3600),
            mip_gap=profile.get('mip_gap', 0.01),
            threads=profile.get('threads', 0),
            symmetry=profile.get('symmetry', 0),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'name': self.name,
            'time_limit': self.time_limit,
            'mip_gap': self.mip_gap,
            'threads': self.threads,
            'symmetry': self.symmetry,
            'log_to_console': self.log_to_console,
            'log_file': str(self.log_file) if self.log_file else None,
            'extra_params': self.extra_params
        }


@dataclass
class ModelConfig:
    """Configuration for model building."""
    
    formulation: str = "cortes2010"  # cortes2010, rais2014, lyu2023
    big_m_strategy: str = "auto"  # auto, manual
    big_m_value: float = 999.0  # Used when strategy = manual
    big_m_multiplier: float = 1.5  # Used when strategy = auto
    
    @classmethod
    def from_yaml(cls, filepath: Path | str) -> 'ModelConfig':
        """Load model configuration from YAML file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        model_config = config_dict.get('model', {})
        
        return cls(
            formulation=model_config.get('formulation', 'cortes2010'),
            big_m_strategy=model_config.get('big_m_strategy', 'auto'),
            big_m_value=model_config.get('big_m_value', 999.0),
            big_m_multiplier=model_config.get('big_m_multiplier', 1.5)
        )
