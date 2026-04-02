"""
Data preprocessor for PDPTWT instances.

This module provides functions to preprocess problem instances:
- Compute distance matrices
- Group nodes by type
- Generate valid arcs for different model formulations
"""

import math
from typing import List, Tuple, Dict
from loguru import logger

from .schemas import ProblemInstance, Node, DistanceMatrix, NodeGroups


def compute_distance_matrix(instance: ProblemInstance) -> DistanceMatrix:
    """
    Compute Euclidean distance matrix for all nodes.
    
    Args:
        instance: ProblemInstance with nodes
        
    Returns:
        DistanceMatrix with precomputed distances
    """
    logger.info(f"Computing distance matrix for {len(instance.nodes)} nodes")
    
    distances: Dict[str, Dict[str, float]] = {}
    
    for node1 in instance.nodes:
        distances[node1.name] = {}
        for node2 in instance.nodes:
            if node1.name != node2.name:
                dist = _euclidean_distance(node1, node2)
                distances[node1.name][node2.name] = dist
    
    distance_matrix = DistanceMatrix(distances=distances)
    logger.success(f"Computed {len(distances)} x {len(distances)} distance matrix")
    
    return distance_matrix


def _euclidean_distance(node1: Node, node2: Node) -> float:
    """Calculate Euclidean distance between two nodes."""
    return math.sqrt((node2.x - node1.x) ** 2 + (node2.y - node1.y) ** 2)


def group_nodes(instance: ProblemInstance) -> NodeGroups:
    """
    Group nodes by type for easy access in model building.
    
    Args:
        instance: ProblemInstance with nodes
        
    Returns:
        NodeGroups with categorized node lists
    """
    logger.info(f"Grouping nodes for instance {instance.name}")
    
    groups = NodeGroups(
        all_nodes=[node.name for node in instance.nodes],
        pickup_nodes=[node.name for node in instance.nodes if node.name.startswith('p')],
        delivery_nodes=[node.name for node in instance.nodes if node.name.startswith('d')],
        vehicle_origins=[node.name for node in instance.nodes if node.name.startswith('o')],
        vehicle_destinations=[node.name for node in instance.nodes if node.name.startswith('e')],
        transfer_nodes=[node.name for node in instance.nodes if node.name.startswith('t')]
    )
    
    logger.debug(
        f"Grouped nodes: {len(groups.pickup_nodes)} pickups, "
        f"{len(groups.delivery_nodes)} deliveries, "
        f"{len(groups.vehicle_origins)} origins, "
        f"{len(groups.vehicle_destinations)} destinations, "
        f"{len(groups.transfer_nodes)} transfers"
    )
    
    return groups


def generate_arcs_cortes(instance: ProblemInstance, node_groups: NodeGroups) -> List[Tuple[str, str]]:
    """
    Generate valid arcs for Cortes 2010 formulation.
    
    This formulation splits transfer nodes into 'ts' (start) and 'tf' (finish).
    
    Args:
        instance: ProblemInstance
        node_groups: Grouped nodes
        
    Returns:
        List of valid (from_node, to_node) arcs
    """
    logger.info("Generating arcs for Cortes 2010 formulation")
    
    arcs = []
    
    # Expand transfer nodes into ts and tf
    transfer_start = [f"{t}s" if not t.endswith('s') and not t.endswith('f') else t 
                      for t in node_groups.transfer_nodes if not t.endswith('f')]
    transfer_finish = [f"{t}f" if not t.endswith('f') and not t.endswith('s') else t 
                       for t in node_groups.transfer_nodes if not t.endswith('s')]
    
    # Vehicle origins to pickup nodes, transfer starts, and vehicle destinations
    for vo in node_groups.vehicle_origins:
        for p in node_groups.pickup_nodes:
            arcs.append((vo, p))
        for ts in transfer_start:
            arcs.append((vo, ts))
        # Vehicle can go directly to its own destination
        vd = vo.replace('o', 'e')
        if vd in node_groups.vehicle_destinations:
            arcs.append((vo, vd))
    
    # Pickup nodes to other nodes
    for p in node_groups.pickup_nodes:
        for p2 in node_groups.pickup_nodes:
            if p != p2:
                arcs.append((p, p2))
        for d in node_groups.delivery_nodes:
            # Cannot go directly to own delivery (must visit other nodes first in some formulations)
            arcs.append((p, d))
        for ts in transfer_start:
            arcs.append((p, ts))
    
    # Delivery nodes to other nodes
    for d in node_groups.delivery_nodes:
        for p in node_groups.pickup_nodes:
            # Cannot go from delivery to its own pickup
            if d.replace('d', 'p') != p:
                arcs.append((d, p))
        for d2 in node_groups.delivery_nodes:
            if d != d2:
                arcs.append((d, d2))
        for vd in node_groups.vehicle_destinations:
            arcs.append((d, vd))
        for ts in transfer_start:
            arcs.append((d, ts))
    
    # Transfer start to transfer finish (same location)
    for ts in transfer_start:
        tf = ts.replace('ts', 'tf') if 'ts' in ts else ts.replace('s', 'f')
        if tf in transfer_finish or any(t.startswith(tf.replace('tf', '').replace('f', '')) for t in transfer_finish):
            arcs.append((ts, tf))
    
    # Transfer finish to other nodes
    for tf in transfer_finish:
        for p in node_groups.pickup_nodes:
            arcs.append((tf, p))
        for d in node_groups.delivery_nodes:
            arcs.append((tf, d))
        for vd in node_groups.vehicle_destinations:
            arcs.append((tf, vd))
        for ts in transfer_start:
            # Cannot go back to same transfer
            if not (tf.replace('tf', '').replace('f', '') == ts.replace('ts', '').replace('s', '')):
                arcs.append((tf, ts))
    
    # Remove duplicates
    arcs = list(set(arcs))
    
    logger.success(f"Generated {len(arcs)} arcs for Cortes formulation")
    return arcs


def generate_arcs_standard(instance: ProblemInstance, node_groups: NodeGroups) -> List[Tuple[str, str]]:
    """
    Generate valid arcs for standard formulations (Rais, Lyu).
    
    Args:
        instance: ProblemInstance
        node_groups: Grouped nodes
        
    Returns:
        List of valid (from_node, to_node) arcs
    """
    logger.info("Generating arcs for standard formulation")
    
    arcs = []
    
    # All nodes except self-loops
    for node1 in node_groups.all_nodes:
        for node2 in node_groups.all_nodes:
            if node1 != node2:
                arcs.append((node1, node2))
    
    logger.success(f"Generated {len(arcs)} arcs for standard formulation")
    return arcs


def preprocess_instance(instance: ProblemInstance, compute_distances: bool = True) -> ProblemInstance:
    """
    Preprocess an instance by computing distance matrix and grouping nodes.
    
    Args:
        instance: ProblemInstance to preprocess
        compute_distances: Whether to compute distance matrix
        
    Returns:
        Updated ProblemInstance with distance_matrix and node_groups populated
    """
    logger.info(f"Preprocessing instance {instance.name}")
    
    # Group nodes
    instance.node_groups = group_nodes(instance)
    
    # Compute distance matrix
    if compute_distances:
        instance.distance_matrix = compute_distance_matrix(instance)
    
    logger.success(f"Preprocessing complete for {instance.name}")
    return instance


def calculate_big_m(instance: ProblemInstance, multiplier: float = 1.5) -> float:
    """
    Calculate BigM value based on time windows and distances.
    
    The BigM should be large enough to make constraints non-binding when
    a binary variable is 0, but not so large as to cause numerical issues.
    
    Args:
        instance: ProblemInstance
        multiplier: Safety multiplier (default 1.5)
        
    Returns:
        Calculated BigM value
    """
    if instance.distance_matrix is None:
        instance.distance_matrix = compute_distance_matrix(instance)
    
    # Maximum time window end
    max_time = max(node.b for node in instance.nodes)
    
    # Maximum distance
    max_distance = max(
        instance.distance_matrix.distances[n1][n2]
        for n1 in instance.distance_matrix.distances
        for n2 in instance.distance_matrix.distances[n1]
    )
    
    # BigM should be at least max_time + max_distance
    big_m = (max_time + max_distance) * multiplier
    
    logger.info(f"Calculated BigM = {big_m:.2f} (max_time={max_time}, max_distance={max_distance:.2f})")
    
    return big_m
