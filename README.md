# PDPTWT Models

Models for the **Pickup and Delivery Problem with Time Windows and Transfers (PDPTWT)**.

## Overview

This repository provides a modular, scalable, and maintainable implementation of multiple formulations for the PDPTWT problem. The repository is designed for extensibility, allowing easy addition of new solution methods and problem variants.

## Features

- **Multiple Formulations**: Cortes et al. (2010), Rais et al. (2014), Lyu et al. (2023)
- **Solver Abstraction**: Easy to swap between Gurobi, OR-Tools, CPLEX
- **Type-Safe Data Structures**: Pydantic schemas with validation
- **Configuration Management**: YAML-based configuration with solver profiles
- **Structured Logging**: Comprehensive logging with loguru
- **Robust Error Handling**: Custom exception hierarchy
- **CLI Interface**: Command-line tools for solving and validation
- **Batch Processing**: Solve multiple instances efficiently

## Installation

### Prerequisites

- Python 3.9 or higher
- Gurobi Solver

### Using pip

```bash
pip install -r requirements.txt
```

### Using Poetry (recommended)

```bash
pip install poetry
poetry install
```

## Quick Start

### Solve a Single Instance

```bash
python -m src.cli solve --instance data/PDPTWT/3R-4K-4T-240L-0.txt --model cortes2010
```

### Batch Solve Multiple Instances

```bash
python -m src.cli batch --directory data/PDPTWT --model cortes2010 --profile quick --output results
```

### Validate an Instance

```bash
python -m src.cli validate --instance data/PDPTWT/3R-4K-4T-240L-0.txt
```

## Programmatic API

```python
from pathlib import Path
from src.data.loader import load_instance
from src.data.preprocessor import preprocess_instance
from src.models.cortes2010 import CortesModel
from src.solvers.gurobi_solver import GurobiSolver
from src.solvers.solver_config import SolverConfig, ModelConfig

# Load and preprocess instance
instance = load_instance("data/PDPTWT/3R-4K-4T-240L-0.txt")
preprocess_instance(instance)

# Configure solver
solver_config = SolverConfig(time_limit=3600, mip_gap=0.01)
model_config = ModelConfig(formulation="cortes2010")

# Create model and solver
model = CortesModel(model_config)
solver = GurobiSolver(solver_config)

# Build and solve
solver.create_model("PDPTWT")
model.build_model(solver, instance)
status = solver.optimize()

# Get results
if status.value in ['optimal', 'feasible']:
    obj_value = solver.get_objective_value()
    solve_time = solver.get_solve_time()
    print(f"Objective: {obj_value:.2f}, Time: {solve_time:.2f}s")
```

## Configuration

### Solver Profiles

Pre-defined solver profiles are available in `config/solver_profiles.yaml`:

- **quick**: Fast solving for testing (5 min, 5% gap)
- **standard**: Balanced solving (1 hour, 1% gap)
- **thorough**: Thorough solving (2 hours, 0.1% gap)
- **production**: Production settings (30 min, 2% gap)

Use a profile:

```bash
python -m src.cli solve --instance data/PDPTWT/3R-4K-4T-240L-0.txt --profile quick
```

### Custom Configuration

Create a custom YAML configuration file:

```yaml
solver:
  name: "gurobi"
  time_limit: 7200
  mip_gap: 0.005
  threads: 8
  symmetry: 2

model:
  formulation: "cortes2010"
  big_m_strategy: "auto"
  big_m_multiplier: 1.5

logging:
  level: "INFO"
  console: true
  file: "logs/solver.log"
```

Use custom config:

```bash
python -m src.cli solve --instance data/PDPTWT/3R-4K-4T-240L-0.txt --config my_config.yaml
```

## Architecture

```
src/
├── data/           # Data loading, validation, preprocessing
├── models/         # Mathematical formulations
├── solvers/        # Solver abstraction layer
├── postprocessing/ # Solution extraction and visualization
├── utils/          # Logging, exceptions, callbacks
└── cli.py          # Command-line interface
```

### Key Design Principles

1. **Separation of Concerns**: Data, model, solver, and post-processing are strictly decoupled
2. **Solver Agnostic**: Models don't depend on specific solver APIs
3. **Type Safety**: Pydantic schemas catch errors at load time
4. **Configuration Over Code**: Parameters externalized to YAML files
5. **Extensibility**: Easy to add new formulations and solvers

## Adding New Formulations

To add a new model formulation:

1. Create a new file in `src/models/` (e.g., `mymodel.py`)
2. Inherit from `PDPTWTModel` base class
3. Implement required methods:
   - `get_name()`: Return model identifier
   - `build_variables()`: Define decision variables
   - `build_objective()`: Define objective function
   - `build_constraints()`: Define constraints

Example:

```python
from src.models.base import PDPTWTModel

class MyModel(PDPTWTModel):
    def get_name(self) -> str:
        return "MyModel2024"
    
    def build_variables(self, solver, instance):
        # Define variables
        pass
    
    def build_objective(self, solver, instance, variables):
        # Define objective
        pass
    
    def build_constraints(self, solver, instance, variables):
        # Define constraints
        pass
```

## Adding New Solvers

To add support for a new solver (e.g., OR-Tools):

1. Create `src/solvers/ortools_solver.py`
2. Inherit from `BaseSolver`
3. Implement all abstract methods
4. Map solver-specific types to the abstract interface

The abstract interface ensures models work with any solver without modification.

## Data Format

Instance files use tab-separated format:

```
nr	nv	nt	capacity
3	4	4	99

node	x	y	a	b	load
p0	48	99	0	107	1
d0	0	45	73	180	-1
o0	30	30	0	240	0
e0	30	30	0	240	0
t0	60	90	0	240	0
...
```

## Testing

Run tests with pytest:

```bash
pytest tests/ --cov=src --cov-report=html
```

## Project Structure

```
.
├── config/                 # Configuration files
├── data/                   # Instance data
│   ├── PDPT/              # PDPT instances
│   └── PDPTWT/            # PDPTWT instances
├── src/                    # Source code
├── tests/                  # Test suite
├── results/                # Output directory
├── logs/                   # Log files
├── main.py                 # Batch processing entry point
├── pyproject.toml          # Project configuration
└── requirements.txt        # Dependencies
```

## Contributing

When adding new features:

1. Follow the existing architecture patterns
2. Add type hints to all functions
3. Write unit tests for new functionality
4. Update documentation
5. Use structured logging

## License

[Add your license here]

## Citation

If you use this solver in your research, please cite the original papers:

- Cortés, C. E., Matamala, M., & Contardo, C. (2010). The pickup and delivery problem with transfers: Formulation and a branch-and-cut solution method. European Journal of Operational Research, 200(3), 711-724.

- Rais, A., Alvelos, F., & Carvalho, M. S. (2014). New mixed integer-programming model for the pickup-and-delivery problem with transshipment. European Journal of Operational Research, 235(3), 530-539.

- Lyu, Z., & Yu, V. F. (2023). An exact algorithm for the pickup and delivery problem with time windows and transfers. Computers & Operations Research, 150, 106080.

## Contact

[Add your contact information]
