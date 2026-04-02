"""
Data loader for PDPTWT instances.

This module provides functions to load problem instances from text files
and return validated Pydantic objects.
"""

from pathlib import Path
from typing import List, Optional
import pandas as pd
from loguru import logger

from .schemas import ProblemInstance, ProblemMetadata, Node


class DataLoadError(Exception):
    """Exception raised when data loading fails."""
    pass


def load_instance(filepath: Path | str, validate: bool = True) -> ProblemInstance:
    """
    Load a PDPTWT instance from a text file.
    
    Args:
        filepath: Path to the instance file
        validate: Whether to validate the instance after loading
        
    Returns:
        ProblemInstance object with validated data
        
    Raises:
        DataLoadError: If file cannot be read or data is invalid
        FileNotFoundError: If file does not exist
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Instance file not found: {filepath}")
    
    logger.info(f"Loading instance from {filepath}")
    
    try:
        # Read metadata (first 2 lines)
        metadata_df = pd.read_csv(
            filepath,
            nrows=1,
            sep='\t',
            on_bad_lines='skip'
        )
        
        # Extract metadata
        metadata = ProblemMetadata(
            nr=int(metadata_df['nr'].iloc[0]),
            nv=int(metadata_df['nv'].iloc[0]),
            nt=int(metadata_df['nt'].iloc[0]),
            capacity=int(metadata_df['capacity'].iloc[0])
        )
        
        logger.debug(f"Metadata: nr={metadata.nr}, nv={metadata.nv}, nt={metadata.nt}, capacity={metadata.capacity}")
        
        # Read node data (skip first 3 lines: header, metadata, blank line)
        nodes_df = pd.read_csv(filepath, skiprows=3, sep='\t')
        
        # Convert to Node objects
        nodes = []
        for _, row in nodes_df.iterrows():
            node = Node(
                name=str(row['node']),
                x=float(row['x']),
                y=float(row['y']),
                a=float(row['a']),
                b=float(row['b']),
                load=int(row['load'])
            )
            nodes.append(node)
        
        logger.debug(f"Loaded {len(nodes)} nodes")
        
        # Create instance
        instance = ProblemInstance(
            name=filepath.stem,
            metadata=metadata,
            nodes=nodes
        )
        
        logger.success(f"Successfully loaded instance {instance.name}")
        return instance
        
    except Exception as e:
        logger.error(f"Failed to load instance from {filepath}: {e}")
        raise DataLoadError(f"Failed to load instance: {e}") from e


def load_batch(
    directory: Path | str,
    pattern: str = "*.txt",
    validate: bool = True,
    skip_errors: bool = False
) -> List[ProblemInstance]:
    """
    Load multiple instances from a directory.
    
    Args:
        directory: Directory containing instance files
        pattern: Glob pattern for matching files (default: "*.txt")
        validate: Whether to validate instances after loading
        skip_errors: If True, skip files that fail to load; if False, raise exception
        
    Returns:
        List of ProblemInstance objects
        
    Raises:
        DataLoadError: If directory doesn't exist or no files match pattern
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise DataLoadError(f"Directory not found: {directory}")
    
    if not directory.is_dir():
        raise DataLoadError(f"Path is not a directory: {directory}")
    
    # Find matching files
    files = sorted(directory.glob(pattern))
    
    if not files:
        raise DataLoadError(f"No files matching pattern '{pattern}' found in {directory}")
    
    logger.info(f"Found {len(files)} files matching pattern '{pattern}' in {directory}")
    
    instances = []
    errors = []
    
    for filepath in files:
        try:
            instance = load_instance(filepath, validate=validate)
            instances.append(instance)
        except Exception as e:
            error_msg = f"Failed to load {filepath.name}: {e}"
            errors.append(error_msg)
            
            if skip_errors:
                logger.warning(error_msg)
            else:
                logger.error(error_msg)
                raise DataLoadError(error_msg) from e
    
    if errors and skip_errors:
        logger.warning(f"Skipped {len(errors)} files due to errors")
    
    logger.success(f"Successfully loaded {len(instances)} instances")
    return instances


def save_instance(instance: ProblemInstance, filepath: Path | str) -> None:
    """
    Save a ProblemInstance to a text file in the standard format.
    
    Args:
        instance: ProblemInstance to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving instance to {filepath}")
    
    try:
        with open(filepath, 'w') as f:
            # Write metadata header
            f.write("nr\tnv\tnt\tcapacity\n")
            f.write(f"{instance.metadata.nr}\t{instance.metadata.nv}\t{instance.metadata.nt}\t{instance.metadata.capacity}\n")
            f.write("\n")
            
            # Write node header
            f.write("node\tx\ty\ta\tb\tload\n")
            
            # Write nodes
            for node in instance.nodes:
                f.write(f"{node.name}\t{node.x}\t{node.y}\t{node.a}\t{node.b}\t{node.load}\n")
        
        logger.success(f"Successfully saved instance to {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save instance to {filepath}: {e}")
        raise DataLoadError(f"Failed to save instance: {e}") from e
