"""
Main entry point for PDPTWT solver.

This module provides a programmatic API for batch processing instances.
"""

from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
import pandas as pd

from src.data.loader import load_batch
from src.data.preprocessor import preprocess_instance
from src.models.cortes2010 import CortesModel
from src.solvers.gurobi_solver import GurobiSolver
from src.solvers.solver_config import SolverConfig, ModelConfig
from src.utils.logging_config import setup_logging
from src.utils.callbacks import create_callback
from src.data.schemas import SolverStatus


def solve_batch(
    data_dir: str | Path,
    pattern: str = "*.txt",
    model_name: str = "cortes2010",
    solver_config: SolverConfig | None = None,
    model_config: ModelConfig | None = None,
    output_dir: str | Path | None = None
) -> List[Dict[str, Any]]:
    """
    Solve multiple PDPTWT instances.
    
    Args:
        data_dir: Directory containing instance files
        pattern: File pattern to match
        model_name: Model formulation to use
        solver_config: Solver configuration
        model_config: Model configuration
        output_dir: Directory to save results
        
    Returns:
        List of result dictionaries
    """
    # Setup logging
    setup_logging(level="INFO", console=True)
    
    # Load instances
    logger.info(f"Loading instances from {data_dir}")
    instances = load_batch(data_dir, pattern=pattern, skip_errors=True)
    logger.info(f"Loaded {len(instances)} instances")
    
    # Use default configs if not provided
    solver_config = solver_config or SolverConfig()
    model_config = model_config or ModelConfig()
    
    # Solve each instance
    results = []
    
    for instance in instances:
        logger.info(f"\n{'='*60}")
        logger.info(f"Solving: {instance.name}")
        logger.info(f"{'='*60}")
        
        try:
            # Preprocess
            preprocess_instance(instance)
            
            # Create model
            if model_name == "cortes2010":
                model = CortesModel(model_config)
            elif model_name == "rais2014":
                from src.models.rais2014 import RaisModel
                model = RaisModel(model_config)
            elif model_name == "lyu2023":
                from src.models.lyu2023 import LyuModel
                model = LyuModel(model_config)
            else:
                raise ValueError(f"Unknown model: {model_name}. Available: cortes2010, rais2014, lyu2023")
            
            # Solve
            with GurobiSolver(solver_config) as solver:
                solver.create_model(f"PDPTWT_{instance.name}")
                model.build_model(solver, instance)
                
                callback = create_callback(log_interval=30.0)
                status = solver.optimize(callback)
                
                # Collect results
                result = {
                    'instance': instance.name,
                    'status': status.value,
                    'objective': None,
                    'time': solver.get_solve_time(),
                    'gap': None
                }
                
                if status in [SolverStatus.OPTIMAL, SolverStatus.FEASIBLE, SolverStatus.TIME_LIMIT]:
                    result['objective'] = solver.get_objective_value()
                    result['gap'] = solver.get_mip_gap()
                    
                    logger.success(
                        f"Solved {instance.name}: "
                        f"obj={result['objective']:.2f}, "
                        f"time={result['time']:.2f}s"
                    )
                else:
                    logger.warning(f"Instance {instance.name} terminated with status: {status.value}")
                
                results.append(result)
        
        except Exception as e:
            logger.error(f"Error solving {instance.name}: {e}")
            results.append({
                'instance': instance.name,
                'status': 'error',
                'objective': None,
                'time': None,
                'gap': None
            })
    
    # Save results if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(results)
        csv_file = output_path / f"results_{model_name}.csv"
        df.to_csv(csv_file, index=False)
        
        logger.success(f"Results saved to {csv_file}")
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total instances: {len(instances)}")
    logger.info(f"Solved: {sum(1 for r in results if r['status'] in ['optimal', 'feasible'])}")
    logger.info(f"Time limit: {sum(1 for r in results if r['status'] == 'time_limit')}")
    logger.info(f"Errors: {sum(1 for r in results if r['status'] in ['error', 'infeasible'])}")
    
    return results


if __name__ == "__main__":
    # Example usage
    results = solve_batch(
        data_dir="./data/PDPTWT",
        pattern="*.txt",
        model_name="cortes2010",
        output_dir="./results"
    )
