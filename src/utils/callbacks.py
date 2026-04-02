"""
Gurobi callbacks for tracking optimization progress.

This module provides callback functions to monitor solver progress
and collect data for visualization.
"""

import time
from typing import List, Dict, Any
import gurobipy as gp
from loguru import logger

from ..data.schemas import OptimizationProgress


class ProgressCallback:
    """
    Callback class to track optimization progress.
    
    This callback collects objective value, bound, and gap information
    during the optimization process.
    """
    
    def __init__(self, log_interval: float = 10.0):
        """
        Initialize callback.
        
        Args:
            log_interval: Minimum seconds between log messages
        """
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.log_interval = log_interval
        
        self.data: List[OptimizationProgress] = []
        self.best_obj = float('inf')
        self.best_bound = float('-inf')
        self.best_gap = 100.0
    
    def __call__(self, model: gp.Model, where: int) -> None:
        """
        Callback function called by Gurobi during optimization.
        
        Args:
            model: Gurobi model
            where: Callback location code
        """
        if where == gp.GRB.Callback.MIP:
            # Get current values
            try:
                cur_obj = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
                cur_bound = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
                
                # Calculate gap
                if abs(cur_obj) > 1e-10:
                    gap = abs(cur_bound - cur_obj) / abs(cur_obj) * 100
                else:
                    gap = 100.0
                
                # Check if values changed
                if cur_obj != self.best_obj or cur_bound != self.best_bound:
                    self.best_obj = cur_obj
                    self.best_bound = cur_bound
                    self.best_gap = gap
                    
                    # Record progress
                    elapsed = time.time() - self.start_time
                    progress = OptimizationProgress(
                        timestamp=elapsed,
                        objective_value=cur_obj,
                        objective_bound=cur_bound,
                        gap=gap
                    )
                    self.data.append(progress)
                    
                    # Log periodically
                    if time.time() - self.last_log_time >= self.log_interval:
                        logger.info(
                            f"Progress: {elapsed:.1f}s | "
                            f"Obj: {cur_obj:.2f} | "
                            f"Bound: {cur_bound:.2f} | "
                            f"Gap: {gap:.2f}%"
                        )
                        self.last_log_time = time.time()
            
            except Exception as e:
                logger.warning(f"Callback error: {e}")
    
    def get_progress_data(self) -> List[Dict[str, Any]]:
        """
        Get progress data as list of dictionaries.
        
        Returns:
            List of progress records
        """
        return [
            {
                'time': p.timestamp,
                'objective': p.objective_value,
                'bound': p.objective_bound,
                'gap': p.gap
            }
            for p in self.data
        ]
    
    def get_final_gap(self) -> float:
        """Get final MIP gap."""
        return self.best_gap
    
    def get_solve_time(self) -> float:
        """Get total solve time."""
        return time.time() - self.start_time


def create_callback(log_interval: float = 10.0) -> ProgressCallback:
    """
    Create a progress callback instance.
    
    Args:
        log_interval: Minimum seconds between log messages
        
    Returns:
        ProgressCallback instance
    """
    return ProgressCallback(log_interval=log_interval)
