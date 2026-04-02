"""
Abstract base class for optimization solvers.

This module defines the interface that all solver implementations must follow,
enabling easy swapping between different solvers (Gurobi, OR-Tools, CPLEX, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from enum import Enum

from ..data.schemas import ProblemInstance, Solution, SolverStatus
from .solver_config import SolverConfig


class VarType(str, Enum):
    """Variable types."""
    BINARY = "binary"
    INTEGER = "integer"
    CONTINUOUS = "continuous"


class ObjectiveSense(str, Enum):
    """Optimization sense."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class BaseSolver(ABC):
    """
    Abstract base class for optimization solvers.
    
    This interface allows different solver backends (Gurobi, OR-Tools, CPLEX)
    to be used interchangeably without modifying model code.
    """
    
    def __init__(self, config: Optional[SolverConfig] = None):
        """
        Initialize solver.
        
        Args:
            config: Solver configuration
        """
        self.config = config or SolverConfig()
        self.model: Any = None
        self.variables: Dict[str, Any] = {}
        self.constraints: Dict[str, Any] = {}
    
    @abstractmethod
    def create_model(self, name: str = "PDPTWT") -> None:
        """
        Create a new optimization model.
        
        Args:
            name: Model name
        """
        pass
    
    @abstractmethod
    def add_variable(
        self,
        name: str,
        var_type: VarType,
        lb: float = 0.0,
        ub: float = float('inf'),
        **kwargs
    ) -> Any:
        """
        Add a decision variable to the model.
        
        Args:
            name: Variable name
            var_type: Variable type (binary, integer, continuous)
            lb: Lower bound
            ub: Upper bound
            **kwargs: Additional solver-specific parameters
            
        Returns:
            Variable object
        """
        pass
    
    @abstractmethod
    def add_variables(
        self,
        indices: list,
        name: str,
        var_type: VarType,
        lb: float = 0.0,
        ub: float = float('inf'),
        **kwargs
    ) -> Dict[Any, Any]:
        """
        Add multiple indexed variables to the model.
        
        Args:
            indices: List of indices for variables
            name: Base variable name
            var_type: Variable type
            lb: Lower bound
            ub: Upper bound
            **kwargs: Additional solver-specific parameters
            
        Returns:
            Dictionary mapping indices to variable objects
        """
        pass
    
    @abstractmethod
    def add_constraint(self, expression: Any, name: str) -> None:
        """
        Add a constraint to the model.
        
        Args:
            expression: Constraint expression
            name: Constraint name
        """
        pass
    
    @abstractmethod
    def set_objective(self, expression: Any, sense: ObjectiveSense) -> None:
        """
        Set the objective function.
        
        Args:
            expression: Objective expression
            sense: Optimization sense (minimize/maximize)
        """
        pass
    
    @abstractmethod
    def optimize(self, callback: Optional[Any] = None) -> SolverStatus:
        """
        Solve the optimization model.
        
        Args:
            callback: Optional callback function for progress tracking
            
        Returns:
            Solver termination status
        """
        pass
    
    @abstractmethod
    def get_objective_value(self) -> float:
        """
        Get the objective function value.
        
        Returns:
            Objective value
        """
        pass
    
    @abstractmethod
    def get_variable_value(self, variable: Any) -> float:
        """
        Get the value of a variable in the solution.
        
        Args:
            variable: Variable object
            
        Returns:
            Variable value
        """
        pass
    
    @abstractmethod
    def get_solve_time(self) -> float:
        """
        Get the solve time in seconds.
        
        Returns:
            Solve time
        """
        pass
    
    @abstractmethod
    def get_mip_gap(self) -> Optional[float]:
        """
        Get the final MIP gap.
        
        Returns:
            MIP gap (None if not applicable)
        """
        pass
    
    @abstractmethod
    def dispose(self) -> None:
        """Clean up solver resources."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.dispose()
