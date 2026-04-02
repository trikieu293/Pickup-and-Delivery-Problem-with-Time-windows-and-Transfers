"""
Gurobi solver implementation.

This module provides a Gurobi-specific implementation of the BaseSolver interface.
"""

from typing import Any, Dict, Optional
import gurobipy as gp
from gurobipy import GRB
from loguru import logger

from .base import BaseSolver, VarType, ObjectiveSense
from .solver_config import SolverConfig
from ..data.schemas import SolverStatus
from ..utils.exceptions import (
    SolverError,
    ModelInfeasibleError,
    ModelUnboundedError,
    SolverTimeoutError,
    SolverConfigError
)


class GurobiSolver(BaseSolver):
    """
    Gurobi solver implementation.
    
    This class wraps Gurobi's API to conform to the BaseSolver interface,
    allowing models to be solver-agnostic.
    """
    
    def __init__(self, config: Optional[SolverConfig] = None):
        """
        Initialize Gurobi solver.
        
        Args:
            config: Solver configuration
        """
        super().__init__(config)
        self.model: Optional[gp.Model] = None
        self._status: Optional[SolverStatus] = None
    
    def create_model(self, name: str = "PDPTWT") -> None:
        """
        Create a new Gurobi model.
        
        Args:
            name: Model name
        """
        try:
            self.model = gp.Model(name)
            logger.info(f"Created Gurobi model: {name}")
            
            # Apply configuration
            self._apply_config()
            
        except gp.GurobiError as e:
            logger.error(f"Failed to create Gurobi model: {e}")
            raise SolverConfigError(f"Failed to create model: {e}") from e
    
    def _apply_config(self) -> None:
        """Apply solver configuration to Gurobi model."""
        if self.model is None:
            raise SolverError("Model not created")
        
        try:
            # Time limit
            self.model.Params.TimeLimit = self.config.time_limit
            logger.debug(f"Set TimeLimit = {self.config.time_limit}")
            
            # MIP gap
            self.model.Params.MIPGap = self.config.mip_gap
            logger.debug(f"Set MIPGap = {self.config.mip_gap}")
            
            # Threads
            if self.config.threads > 0:
                self.model.Params.Threads = self.config.threads
                logger.debug(f"Set Threads = {self.config.threads}")
            
            # Symmetry
            self.model.Params.Symmetry = self.config.symmetry
            logger.debug(f"Set Symmetry = {self.config.symmetry}")
            
            # Console logging
            if not self.config.log_to_console:
                self.model.Params.OutputFlag = 0
            
            # Log file
            if self.config.log_file:
                self.model.Params.LogFile = str(self.config.log_file)
                logger.debug(f"Set LogFile = {self.config.log_file}")
            
            # Extra parameters
            for param, value in self.config.extra_params.items():
                setattr(self.model.Params, param, value)
                logger.debug(f"Set {param} = {value}")
            
            self.model.update()
            
        except Exception as e:
            logger.error(f"Failed to apply configuration: {e}")
            raise SolverConfigError(f"Failed to apply config: {e}") from e
    
    def add_variable(
        self,
        name: str,
        var_type: VarType,
        lb: float = 0.0,
        ub: float = float('inf'),
        **kwargs
    ) -> gp.Var:
        """
        Add a decision variable to the model.
        
        Args:
            name: Variable name
            var_type: Variable type
            lb: Lower bound
            ub: Upper bound
            **kwargs: Additional Gurobi parameters
            
        Returns:
            Gurobi variable object
        """
        if self.model is None:
            raise SolverError("Model not created")
        
        # Map VarType to Gurobi type
        gurobi_type = {
            VarType.BINARY: GRB.BINARY,
            VarType.INTEGER: GRB.INTEGER,
            VarType.CONTINUOUS: GRB.CONTINUOUS
        }[var_type]
        
        var = self.model.addVar(
            lb=lb,
            ub=ub,
            vtype=gurobi_type,
            name=name,
            **kwargs
        )
        
        self.variables[name] = var
        return var
    
    def add_variables(
        self,
        indices: list,
        name: str,
        var_type: VarType,
        lb: float = 0.0,
        ub: float = float('inf'),
        **kwargs
    ) -> Dict[Any, gp.Var]:
        """
        Add multiple indexed variables to the model.
        
        Args:
            indices: List of indices for variables
            name: Base variable name
            var_type: Variable type
            lb: Lower bound
            ub: Upper bound
            **kwargs: Additional Gurobi parameters
            
        Returns:
            Dictionary mapping indices to Gurobi variables
        """
        if self.model is None:
            raise SolverError("Model not created")
        
        # Map VarType to Gurobi type
        gurobi_type = {
            VarType.BINARY: GRB.BINARY,
            VarType.INTEGER: GRB.INTEGER,
            VarType.CONTINUOUS: GRB.CONTINUOUS
        }[var_type]
        
        vars_dict = self.model.addVars(
            indices,
            lb=lb,
            ub=ub,
            vtype=gurobi_type,
            name=name,
            **kwargs
        )
        
        self.variables[name] = vars_dict
        return vars_dict
    
    def add_constraint(self, expression: Any, name: str) -> None:
        """
        Add a constraint to the model.
        
        Args:
            expression: Gurobi constraint expression
            name: Constraint name
        """
        if self.model is None:
            raise SolverError("Model not created")
        
        constr = self.model.addConstr(expression, name=name)
        self.constraints[name] = constr
    
    def set_objective(self, expression: Any, sense: ObjectiveSense) -> None:
        """
        Set the objective function.
        
        Args:
            expression: Gurobi objective expression
            sense: Optimization sense
        """
        if self.model is None:
            raise SolverError("Model not created")
        
        gurobi_sense = GRB.MINIMIZE if sense == ObjectiveSense.MINIMIZE else GRB.MAXIMIZE
        self.model.setObjective(expression, gurobi_sense)
        logger.info(f"Set objective with sense: {sense.value}")
    
    def optimize(self, callback: Optional[Any] = None) -> SolverStatus:
        """
        Solve the optimization model.
        
        Args:
            callback: Optional callback function for progress tracking
            
        Returns:
            Solver termination status
        """
        if self.model is None:
            raise SolverError("Model not created")
        
        logger.info("Starting optimization...")
        
        try:
            # Update model before solving
            self.model.update()
            
            # Optimize
            if callback:
                self.model.optimize(callback)
            else:
                self.model.optimize()
            
            # Map Gurobi status to SolverStatus
            self._status = self._map_status(self.model.Status)
            
            logger.info(f"Optimization finished with status: {self._status.value}")
            
            return self._status
            
        except gp.GurobiError as e:
            logger.error(f"Gurobi error during optimization: {e}")
            raise SolverError(f"Optimization failed: {e}") from e
    
    def _map_status(self, gurobi_status: int) -> SolverStatus:
        """Map Gurobi status code to SolverStatus."""
        status_map = {
            GRB.OPTIMAL: SolverStatus.OPTIMAL,
            GRB.INFEASIBLE: SolverStatus.INFEASIBLE,
            GRB.INF_OR_UNBD: SolverStatus.UNBOUNDED,
            GRB.UNBOUNDED: SolverStatus.UNBOUNDED,
            GRB.TIME_LIMIT: SolverStatus.TIME_LIMIT,
        }
        
        return status_map.get(gurobi_status, SolverStatus.ERROR)
    
    def get_objective_value(self) -> float:
        """
        Get the objective function value.
        
        Returns:
            Objective value
            
        Raises:
            SolverError: If no solution available
        """
        if self.model is None:
            raise SolverError("Model not created")
        
        try:
            if self.model.SolCount > 0:
                return self.model.ObjVal
            else:
                raise SolverError("No solution available")
        except gp.GurobiError as e:
            raise SolverError(f"Failed to get objective value: {e}") from e
    
    def get_variable_value(self, variable: gp.Var) -> float:
        """
        Get the value of a variable in the solution.
        
        Args:
            variable: Gurobi variable object
            
        Returns:
            Variable value
        """
        if self.model is None:
            raise SolverError("Model not created")
        
        try:
            return variable.X
        except gp.GurobiError as e:
            raise SolverError(f"Failed to get variable value: {e}") from e
    
    def get_solve_time(self) -> float:
        """
        Get the solve time in seconds.
        
        Returns:
            Solve time
        """
        if self.model is None:
            raise SolverError("Model not created")
        
        return self.model.Runtime
    
    def get_mip_gap(self) -> Optional[float]:
        """
        Get the final MIP gap.
        
        Returns:
            MIP gap (None if not applicable)
        """
        if self.model is None:
            raise SolverError("Model not created")
        
        try:
            if self.model.SolCount > 0:
                return self.model.MIPGap
            else:
                return None
        except gp.GurobiError:
            return None
    
    def dispose(self) -> None:
        """Clean up Gurobi model resources."""
        if self.model is not None:
            self.model.dispose()
            self.model = None
            logger.debug("Disposed Gurobi model")
    
    def get_model(self) -> gp.Model:
        """
        Get the underlying Gurobi model.
        
        Returns:
            Gurobi model object
        """
        if self.model is None:
            raise SolverError("Model not created")
        return self.model
