"""
Abstract base class for PDPTWT models.

This module defines the interface that all model formulations must follow,
enabling consistent model building across different formulations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from loguru import logger

from ..data.schemas import ProblemInstance
from ..solvers.base import BaseSolver
from ..solvers.solver_config import ModelConfig


class PDPTWTModel(ABC):
    """
    Abstract base class for PDPTWT model formulations.
    
    This interface ensures all model implementations follow the same structure,
    making it easy to swap between different formulations (Cortes, Rais, Lyu, etc.).
    """
    
    def __init__(self, config: ModelConfig | None = None):
        """
        Initialize model.
        
        Args:
            config: Model configuration
        """
        self.config = config or ModelConfig()
        self.instance: ProblemInstance | None = None
        self.solver: BaseSolver | None = None
        self.variables: Dict[str, Any] = {}
        self.big_m: float = 999.0
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the model name/identifier.
        
        Returns:
            Model name (e.g., "Cortes2010", "Rais2014")
        """
        pass
    
    @abstractmethod
    def build_variables(self, solver: BaseSolver, instance: ProblemInstance) -> Dict[str, Any]:
        """
        Define decision variables for the model.
        
        Args:
            solver: Solver instance
            instance: Problem instance
            
        Returns:
            Dictionary of variables
        """
        pass
    
    @abstractmethod
    def build_objective(self, solver: BaseSolver, instance: ProblemInstance, variables: Dict[str, Any]) -> None:
        """
        Define the objective function.
        
        Args:
            solver: Solver instance
            instance: Problem instance
            variables: Dictionary of decision variables
        """
        pass
    
    @abstractmethod
    def build_constraints(self, solver: BaseSolver, instance: ProblemInstance, variables: Dict[str, Any]) -> None:
        """
        Define all constraints for the model.
        
        Args:
            solver: Solver instance
            instance: Problem instance
            variables: Dictionary of decision variables
        """
        pass
    
    def build_model(self, solver: BaseSolver, instance: ProblemInstance) -> None:
        """
        Build the complete optimization model.
        
        This method orchestrates the model building process by calling
        build_variables, build_objective, and build_constraints in sequence.
        
        Args:
            solver: Solver instance
            instance: Problem instance
        """
        self.solver = solver
        self.instance = instance
        
        logger.info(f"Building {self.get_name()} model for instance {instance.name}")
        
        # Calculate BigM if using auto strategy
        if self.config.big_m_strategy == "auto":
            from ..data.preprocessor import calculate_big_m
            self.big_m = calculate_big_m(instance, self.config.big_m_multiplier)
        else:
            self.big_m = self.config.big_m_value
        
        logger.info(f"Using BigM = {self.big_m}")
        
        # Build model components
        logger.info("Building variables...")
        self.variables = self.build_variables(solver, instance)
        
        logger.info("Building objective function...")
        self.build_objective(solver, instance, self.variables)
        
        logger.info("Building constraints...")
        self.build_constraints(solver, instance, self.variables)
        
        logger.success(f"Model building complete for {self.get_name()}")
    
    def get_formulation_info(self) -> Dict[str, Any]:
        """
        Get information about the model formulation.
        
        Returns:
            Dictionary with formulation details
        """
        return {
            'name': self.get_name(),
            'big_m': self.big_m,
            'big_m_strategy': self.config.big_m_strategy,
        }
