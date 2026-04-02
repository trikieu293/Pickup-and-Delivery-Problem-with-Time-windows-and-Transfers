"""
Command-line interface for PDPTWT solver.

This module provides a CLI for solving PDPTWT instances using different formulations.
"""

import click
from pathlib import Path
from loguru import logger

from .data.loader import load_instance, load_batch
from .data.validator import validate_instance
from .data.preprocessor import preprocess_instance
from .models.cortes2010 import CortesModel
from .solvers.gurobi_solver import GurobiSolver
from .solvers.solver_config import SolverConfig, ModelConfig
from .utils.logging_config import setup_logging
from .utils.callbacks import create_callback
from .data.schemas import SolverStatus


@click.group()
@click.option('--log-level', default='INFO', help='Logging level')
@click.option('--log-file', type=click.Path(), help='Log file path')
def cli(log_level, log_file):
    """PDPTWT Solver - Production-ready solver for Pickup and Delivery with Time Windows and Transfers."""
    setup_logging(level=log_level, log_file=log_file)


@cli.command()
@click.option('--instance', '-i', required=True, type=click.Path(exists=True), help='Instance file path')
@click.option('--model', '-m', default='cortes2010', help='Model formulation (cortes2010, rais2014, lyu2023)')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file')
@click.option('--profile', '-p', help='Solver profile (quick, standard, thorough)')
@click.option('--output', '-o', type=click.Path(), help='Output directory for results')
def solve(instance, model, config, profile, output):
    """Solve a single PDPTWT instance."""
    
    logger.info(f"Solving instance: {instance}")
    logger.info(f"Using model: {model}")
    
    try:
        # Load instance
        instance_obj = load_instance(instance)
        logger.info(f"Loaded instance: {instance_obj.name}")
        
        # Validate instance
        is_valid, warnings = validate_instance(instance_obj, strict=False)
        if warnings:
            logger.warning(f"Instance has {len(warnings)} validation warnings")
        
        # Preprocess instance
        preprocess_instance(instance_obj)
        
        # Load configuration
        if config:
            solver_config = SolverConfig.from_yaml(config)
            model_config = ModelConfig.from_yaml(config)
        elif profile:
            profiles_file = Path(__file__).parent.parent / 'config' / 'solver_profiles.yaml'
            solver_config = SolverConfig.from_profile(profile, profiles_file)
            model_config = ModelConfig()
        else:
            solver_config = SolverConfig()
            model_config = ModelConfig()
        
        # Create model
        if model == 'cortes2010':
            model_obj = CortesModel(model_config)
        elif model == 'rais2014':
            from .models.rais2014 import RaisModel
            model_obj = RaisModel(model_config)
        elif model == 'lyu2023':
            from .models.lyu2023 import LyuModel
            model_obj = LyuModel(model_config)
        else:
            raise ValueError(f"Unknown model: {model}. Available: cortes2010, rais2014, lyu2023")
        
        # Create solver
        with GurobiSolver(solver_config) as solver:
            solver.create_model(f"PDPTWT_{instance_obj.name}")
            
            # Build model
            model_obj.build_model(solver, instance_obj)
            
            # Create callback
            callback = create_callback(log_interval=10.0)
            
            # Solve
            status = solver.optimize(callback)
            
            # Report results
            logger.info(f"Solver status: {status.value}")
            
            if status in [SolverStatus.OPTIMAL, SolverStatus.FEASIBLE, SolverStatus.TIME_LIMIT]:
                obj_value = solver.get_objective_value()
                solve_time = solver.get_solve_time()
                mip_gap = solver.get_mip_gap()
                
                logger.success(f"Objective value: {obj_value:.2f}")
                logger.info(f"Solve time: {solve_time:.2f}s")
                if mip_gap is not None:
                    logger.info(f"MIP gap: {mip_gap*100:.2f}%")
                
                # Save results if output specified
                if output:
                    output_path = Path(output)
                    output_path.mkdir(parents=True, exist_ok=True)
                    result_file = output_path / f"{instance_obj.name}_result.txt"
                    
                    with open(result_file, 'w') as f:
                        f.write(f"Instance: {instance_obj.name}\n")
                        f.write(f"Model: {model}\n")
                        f.write(f"Status: {status.value}\n")
                        f.write(f"Objective: {obj_value:.2f}\n")
                        f.write(f"Time: {solve_time:.2f}s\n")
                        if mip_gap is not None:
                            f.write(f"Gap: {mip_gap*100:.2f}%\n")
                    
                    logger.info(f"Results saved to {result_file}")
            else:
                logger.error(f"Solver terminated with status: {status.value}")
                return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Error solving instance: {e}")
        raise


@cli.command()
@click.option('--directory', '-d', required=True, type=click.Path(exists=True), help='Directory with instances')
@click.option('--pattern', default='*.txt', help='File pattern to match')
@click.option('--model', '-m', default='cortes2010', help='Model formulation')
@click.option('--profile', '-p', help='Solver profile')
@click.option('--output', '-o', type=click.Path(), help='Output directory')
def batch(directory, pattern, model, profile, output):
    """Solve multiple instances from a directory."""
    
    logger.info(f"Batch solving instances from: {directory}")
    
    try:
        # Load instances
        instances = load_batch(directory, pattern=pattern, skip_errors=True)
        logger.info(f"Loaded {len(instances)} instances")
        
        # Load configuration
        if profile:
            profiles_file = Path(__file__).parent.parent / 'config' / 'solver_profiles.yaml'
            solver_config = SolverConfig.from_profile(profile, profiles_file)
        else:
            solver_config = SolverConfig()
        
        model_config = ModelConfig()
        
        # Solve each instance
        results = []
        for instance_obj in instances:
            logger.info(f"\n{'='*60}")
            logger.info(f"Solving: {instance_obj.name}")
            logger.info(f"{'='*60}")
            
            try:
                # Preprocess
                preprocess_instance(instance_obj)
                
                # Create model
                if model == 'cortes2010':
                    model_obj = CortesModel(model_config)
                elif model == 'rais2014':
                    from .models.rais2014 import RaisModel
                    model_obj = RaisModel(model_config)
                elif model == 'lyu2023':
                    from .models.lyu2023 import LyuModel
                    model_obj = LyuModel(model_config)
                else:
                    raise ValueError(f"Unknown model: {model}")
                
                # Solve
                with GurobiSolver(solver_config) as solver:
                    solver.create_model(f"PDPTWT_{instance_obj.name}")
                    model_obj.build_model(solver, instance_obj)
                    
                    callback = create_callback(log_interval=30.0)
                    status = solver.optimize(callback)
                    
                    if status in [SolverStatus.OPTIMAL, SolverStatus.FEASIBLE, SolverStatus.TIME_LIMIT]:
                        obj_value = solver.get_objective_value()
                        solve_time = solver.get_solve_time()
                        
                        results.append({
                            'instance': instance_obj.name,
                            'status': status.value,
                            'objective': obj_value,
                            'time': solve_time
                        })
                        
                        logger.success(f"Solved {instance_obj.name}: obj={obj_value:.2f}, time={solve_time:.2f}s")
                    else:
                        results.append({
                            'instance': instance_obj.name,
                            'status': status.value,
                            'objective': None,
                            'time': solver.get_solve_time()
                        })
                        logger.warning(f"Instance {instance_obj.name} terminated with status: {status.value}")
            
            except Exception as e:
                logger.error(f"Error solving {instance_obj.name}: {e}")
                results.append({
                    'instance': instance_obj.name,
                    'status': 'error',
                    'objective': None,
                    'time': None
                })
        
        # Save batch results
        if output:
            import pandas as pd
            output_path = Path(output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            df = pd.DataFrame(results)
            csv_file = output_path / f"batch_results_{model}.csv"
            df.to_csv(csv_file, index=False)
            
            logger.success(f"Batch results saved to {csv_file}")
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("BATCH SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total instances: {len(instances)}")
        logger.info(f"Solved: {sum(1 for r in results if r['status'] in ['optimal', 'feasible'])}")
        logger.info(f"Time limit: {sum(1 for r in results if r['status'] == 'time_limit')}")
        logger.info(f"Errors: {sum(1 for r in results if r['status'] in ['error', 'infeasible'])}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise


@cli.command()
@click.option('--instance', '-i', required=True, type=click.Path(exists=True), help='Instance file path')
def validate(instance):
    """Validate a PDPTWT instance."""
    
    logger.info(f"Validating instance: {instance}")
    
    try:
        # Load instance
        instance_obj = load_instance(instance)
        
        # Validate
        is_valid, warnings = validate_instance(instance_obj, strict=False)
        
        if is_valid:
            logger.success(f"Instance {instance_obj.name} is valid!")
        else:
            logger.warning(f"Instance {instance_obj.name} has validation issues:")
            for warning in warnings:
                logger.warning(f"  - {warning}")
        
        return 0 if is_valid else 1
        
    except Exception as e:
        logger.error(f"Error validating instance: {e}")
        raise


def main():
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main()
