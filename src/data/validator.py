"""
Data validator for PDPTWT instances.

This module provides validation functions to check instance integrity
beyond basic Pydantic validation.
"""

from typing import List, Tuple
from loguru import logger

from .schemas import ProblemInstance, Node


class ValidationError(Exception):
    """Exception raised when validation fails."""
    pass


def validate_instance(instance: ProblemInstance, strict: bool = True) -> Tuple[bool, List[str]]:
    """
    Validate a PDPTWT instance for correctness.
    
    Args:
        instance: ProblemInstance to validate
        strict: If True, raise exception on validation failure
        
    Returns:
        Tuple of (is_valid, list_of_warnings)
        
    Raises:
        ValidationError: If strict=True and validation fails
    """
    warnings = []
    
    logger.info(f"Validating instance {instance.name}")
    
    # Check node counts
    warnings.extend(_validate_node_counts(instance))
    
    # Check time windows
    warnings.extend(_validate_time_windows(instance))
    
    # Check load conservation
    warnings.extend(_validate_load_conservation(instance))
    
    # Check node naming
    warnings.extend(_validate_node_naming(instance))
    
    # Check coordinates
    warnings.extend(_validate_coordinates(instance))
    
    is_valid = len(warnings) == 0
    
    if warnings:
        logger.warning(f"Validation found {len(warnings)} issues:")
        for warning in warnings:
            logger.warning(f"  - {warning}")
        
        if strict:
            raise ValidationError(f"Validation failed with {len(warnings)} issues")
    else:
        logger.success(f"Instance {instance.name} passed all validation checks")
    
    return is_valid, warnings


def _validate_node_counts(instance: ProblemInstance) -> List[str]:
    """Validate that node counts match metadata."""
    warnings = []
    
    pickup_nodes = [n for n in instance.nodes if n.name.startswith('p')]
    delivery_nodes = [n for n in instance.nodes if n.name.startswith('d')]
    vehicle_origins = [n for n in instance.nodes if n.name.startswith('o')]
    vehicle_destinations = [n for n in instance.nodes if n.name.startswith('e')]
    transfer_nodes = [n for n in instance.nodes if n.name.startswith('t') and not n.name.startswith('ts') and not n.name.startswith('tf')]
    
    # Check request nodes
    if len(pickup_nodes) != instance.metadata.nr:
        warnings.append(
            f"Expected {instance.metadata.nr} pickup nodes, found {len(pickup_nodes)}"
        )
    
    if len(delivery_nodes) != instance.metadata.nr:
        warnings.append(
            f"Expected {instance.metadata.nr} delivery nodes, found {len(delivery_nodes)}"
        )
    
    # Check vehicle nodes
    if len(vehicle_origins) != instance.metadata.nv:
        warnings.append(
            f"Expected {instance.metadata.nv} vehicle origin nodes, found {len(vehicle_origins)}"
        )
    
    if len(vehicle_destinations) != instance.metadata.nv:
        warnings.append(
            f"Expected {instance.metadata.nv} vehicle destination nodes, found {len(vehicle_destinations)}"
        )
    
    # Check transfer nodes
    if len(transfer_nodes) != instance.metadata.nt:
        warnings.append(
            f"Expected {instance.metadata.nt} transfer nodes, found {len(transfer_nodes)}"
        )
    
    return warnings


def _validate_time_windows(instance: ProblemInstance) -> List[str]:
    """Validate time window feasibility."""
    warnings = []
    
    for node in instance.nodes:
        # Already checked by Pydantic: a <= b
        
        # Check for negative time windows
        if node.a < 0:
            warnings.append(f"Node {node.name} has negative time window start: {node.a}")
        
        # Check for zero-width time windows (might be intentional)
        if node.a == node.b and node.name.startswith(('p', 'd')):
            warnings.append(f"Node {node.name} has zero-width time window: [{node.a}, {node.b}]")
    
    return warnings


def _validate_load_conservation(instance: ProblemInstance) -> List[str]:
    """Validate load conservation for each request."""
    warnings = []
    
    # Group by request ID
    for i in range(instance.metadata.nr):
        pickup_name = f"p{i}"
        delivery_name = f"d{i}"
        
        try:
            pickup_node = instance.get_node(pickup_name)
            delivery_node = instance.get_node(delivery_name)
            
            # Check that pickup load is positive and delivery load is negative
            if pickup_node.load <= 0:
                warnings.append(f"Pickup node {pickup_name} has non-positive load: {pickup_node.load}")
            
            if delivery_node.load >= 0:
                warnings.append(f"Delivery node {delivery_name} has non-negative load: {delivery_node.load}")
            
            # Check that loads match
            if pickup_node.load != -delivery_node.load:
                warnings.append(
                    f"Load mismatch for request {i}: pickup={pickup_node.load}, delivery={delivery_node.load}"
                )
        
        except KeyError as e:
            warnings.append(f"Missing node for request {i}: {e}")
    
    return warnings


def _validate_node_naming(instance: ProblemInstance) -> List[str]:
    """Validate node naming conventions."""
    warnings = []
    
    for node in instance.nodes:
        # Check that node IDs are sequential
        if node.name.startswith('p'):
            try:
                node_id = int(node.name[1:])
                if node_id >= instance.metadata.nr:
                    warnings.append(f"Pickup node {node.name} has ID >= nr ({instance.metadata.nr})")
            except ValueError:
                warnings.append(f"Invalid pickup node name format: {node.name}")
        
        elif node.name.startswith('d'):
            try:
                node_id = int(node.name[1:])
                if node_id >= instance.metadata.nr:
                    warnings.append(f"Delivery node {node.name} has ID >= nr ({instance.metadata.nr})")
            except ValueError:
                warnings.append(f"Invalid delivery node name format: {node.name}")
        
        elif node.name.startswith('o'):
            try:
                node_id = int(node.name[1:])
                if node_id >= instance.metadata.nv:
                    warnings.append(f"Vehicle origin {node.name} has ID >= nv ({instance.metadata.nv})")
            except ValueError:
                warnings.append(f"Invalid vehicle origin name format: {node.name}")
        
        elif node.name.startswith('e'):
            try:
                node_id = int(node.name[1:])
                if node_id >= instance.metadata.nv:
                    warnings.append(f"Vehicle destination {node.name} has ID >= nv ({instance.metadata.nv})")
            except ValueError:
                warnings.append(f"Invalid vehicle destination name format: {node.name}")
    
    return warnings


def _validate_coordinates(instance: ProblemInstance) -> List[str]:
    """Validate node coordinates."""
    warnings = []
    
    for node in instance.nodes:
        # Check for negative coordinates (might be valid in some cases)
        if node.x < 0 or node.y < 0:
            warnings.append(f"Node {node.name} has negative coordinates: ({node.x}, {node.y})")
    
    # Check for duplicate coordinates (might indicate data issues)
    coords = [(node.x, node.y) for node in instance.nodes]
    unique_coords = set(coords)
    
    if len(coords) != len(unique_coords):
        warnings.append(f"Found {len(coords) - len(unique_coords)} nodes with duplicate coordinates")
    
    return warnings


def validate_capacity_feasibility(instance: ProblemInstance) -> Tuple[bool, List[str]]:
    """
    Check if any single request exceeds vehicle capacity.
    
    Args:
        instance: ProblemInstance to check
        
    Returns:
        Tuple of (is_feasible, list_of_violations)
    """
    violations = []
    
    for node in instance.nodes:
        if node.load > instance.metadata.capacity:
            violations.append(
                f"Node {node.name} has load {node.load} exceeding capacity {instance.metadata.capacity}"
            )
    
    is_feasible = len(violations) == 0
    return is_feasible, violations
