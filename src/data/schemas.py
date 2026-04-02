"""
Pydantic schemas for type-safe data structures in PDPTWT solver.

This module defines the core data structures used throughout the system:
- ProblemMetadata: Instance metadata (nr, nv, nt, capacity)
- Node: Individual node with coordinates, time windows, and load
- ProblemInstance: Complete problem instance
- DistanceMatrix: Precomputed distances between nodes
- Solution: Optimization results
"""

from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
import numpy as np


class NodeType(str, Enum):
    """Node type classification."""
    PICKUP = "pickup"
    DELIVERY = "delivery"
    VEHICLE_ORIGIN = "vehicle_origin"
    VEHICLE_DESTINATION = "vehicle_destination"
    TRANSFER = "transfer"


class SolverStatus(str, Enum):
    """Solver termination status."""
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    TIME_LIMIT = "time_limit"
    ERROR = "error"


class ProblemMetadata(BaseModel):
    """Metadata for a PDPTWT instance."""
    
    nr: int = Field(..., description="Number of requests", gt=0)
    nv: int = Field(..., description="Number of vehicles", gt=0)
    nt: int = Field(..., description="Number of transfer stations", ge=0)
    capacity: int = Field(..., description="Vehicle capacity", gt=0)
    
    @field_validator('nr', 'nv', 'nt', 'capacity')
    @classmethod
    def validate_positive(cls, v: int) -> int:
        """Ensure all values are positive."""
        if v < 0:
            raise ValueError(f"Value must be non-negative, got {v}")
        return v


class Node(BaseModel):
    """A node in the PDPTWT network."""
    
    name: str = Field(..., description="Node identifier (e.g., p0, d0, o0, e0, t0)")
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    a: float = Field(..., description="Time window start (earliest arrival)")
    b: float = Field(..., description="Time window end (latest arrival)")
    load: int = Field(..., description="Load change at node (positive=pickup, negative=delivery)")
    node_type: Optional[NodeType] = Field(None, description="Node type classification")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate node naming convention."""
        if not v or len(v) < 2:
            raise ValueError(f"Invalid node name: {v}")
        prefix = v[0] if v[1].isdigit() else v[:2]
        if prefix not in ['p', 'd', 'o', 'e', 't', 'ts', 'tf']:
            raise ValueError(f"Invalid node prefix: {prefix} in {v}")
        return v
    
    @model_validator(mode='after')
    def validate_time_window(self) -> 'Node':
        """Ensure time window is valid."""
        if self.a > self.b:
            raise ValueError(f"Invalid time window for {self.name}: a={self.a} > b={self.b}")
        return self
    
    @model_validator(mode='after')
    def infer_node_type(self) -> 'Node':
        """Infer node type from name if not provided."""
        if self.node_type is None:
            if self.name.startswith('p'):
                self.node_type = NodeType.PICKUP
            elif self.name.startswith('d'):
                self.node_type = NodeType.DELIVERY
            elif self.name.startswith('o'):
                self.node_type = NodeType.VEHICLE_ORIGIN
            elif self.name.startswith('e'):
                self.node_type = NodeType.VEHICLE_DESTINATION
            elif self.name.startswith('t'):
                self.node_type = NodeType.TRANSFER
        return self


class NodeGroups(BaseModel):
    """Grouped nodes by type for easy access."""
    
    all_nodes: List[str] = Field(default_factory=list)
    pickup_nodes: List[str] = Field(default_factory=list)
    delivery_nodes: List[str] = Field(default_factory=list)
    vehicle_origins: List[str] = Field(default_factory=list)
    vehicle_destinations: List[str] = Field(default_factory=list)
    transfer_nodes: List[str] = Field(default_factory=list)


class DistanceMatrix(BaseModel):
    """Precomputed distance matrix between nodes."""
    
    distances: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Nested dict: distances[from_node][to_node] = distance"
    )
    
    def get_distance(self, from_node: str, to_node: str) -> float:
        """Get distance between two nodes."""
        if from_node not in self.distances:
            raise KeyError(f"Node {from_node} not in distance matrix")
        if to_node not in self.distances[from_node]:
            raise KeyError(f"Node {to_node} not reachable from {from_node}")
        return self.distances[from_node][to_node]
    
    class Config:
        arbitrary_types_allowed = True


class ProblemInstance(BaseModel):
    """Complete PDPTWT problem instance."""
    
    name: str = Field(..., description="Instance name/identifier")
    metadata: ProblemMetadata = Field(..., description="Problem metadata")
    nodes: List[Node] = Field(..., description="List of all nodes")
    distance_matrix: Optional[DistanceMatrix] = Field(None, description="Precomputed distances")
    node_groups: Optional[NodeGroups] = Field(None, description="Grouped nodes by type")
    
    @field_validator('nodes')
    @classmethod
    def validate_nodes(cls, v: List[Node]) -> List[Node]:
        """Validate node list."""
        if not v:
            raise ValueError("Node list cannot be empty")
        
        # Check for duplicate names
        names = [node.name for node in v]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(f"Duplicate node names found: {set(duplicates)}")
        
        return v
    
    @model_validator(mode='after')
    def validate_load_conservation(self) -> 'ProblemInstance':
        """Validate that total pickup load equals total delivery load."""
        pickup_load = sum(node.load for node in self.nodes if node.load > 0)
        delivery_load = sum(abs(node.load) for node in self.nodes if node.load < 0)
        
        if pickup_load != delivery_load:
            raise ValueError(
                f"Load conservation violated: pickup={pickup_load}, delivery={delivery_load}"
            )
        
        return self
    
    def get_node(self, name: str) -> Node:
        """Get node by name."""
        for node in self.nodes:
            if node.name == name:
                return node
        raise KeyError(f"Node {name} not found in instance")


class Route(BaseModel):
    """A vehicle route."""
    
    vehicle_id: int = Field(..., description="Vehicle identifier")
    sequence: List[str] = Field(..., description="Ordered list of node names")
    arrival_times: Dict[str, float] = Field(default_factory=dict, description="Arrival time at each node")
    departure_times: Dict[str, float] = Field(default_factory=dict, description="Departure time at each node")
    load_profile: Dict[str, int] = Field(default_factory=dict, description="Load at each node")
    total_distance: float = Field(0.0, description="Total distance traveled")


class Transfer(BaseModel):
    """A transfer operation between two vehicles."""
    
    request_id: int = Field(..., description="Request being transferred")
    from_vehicle: int = Field(..., description="Source vehicle")
    to_vehicle: int = Field(..., description="Destination vehicle")
    transfer_node: str = Field(..., description="Transfer location")
    time: float = Field(..., description="Transfer time")


class Solution(BaseModel):
    """Complete solution to a PDPTWT instance."""
    
    instance_name: str = Field(..., description="Instance identifier")
    status: SolverStatus = Field(..., description="Solver termination status")
    objective_value: Optional[float] = Field(None, description="Objective function value")
    solve_time: float = Field(..., description="Solve time in seconds")
    mip_gap: Optional[float] = Field(None, description="Final MIP gap (if applicable)")
    routes: List[Route] = Field(default_factory=list, description="Vehicle routes")
    transfers: List[Transfer] = Field(default_factory=list, description="Transfer operations")
    
    # Additional metrics
    total_distance: float = Field(0.0, description="Total distance across all routes")
    num_transfers: int = Field(0, description="Number of transfer operations")
    
    @model_validator(mode='after')
    def compute_metrics(self) -> 'Solution':
        """Compute aggregate metrics from routes and transfers."""
        self.total_distance = sum(route.total_distance for route in self.routes)
        self.num_transfers = len(self.transfers)
        return self


class OptimizationProgress(BaseModel):
    """Optimization progress data for callbacks."""
    
    timestamp: float = Field(..., description="Time since optimization start")
    objective_value: float = Field(..., description="Current best objective value")
    objective_bound: float = Field(..., description="Current objective bound")
    gap: float = Field(..., description="MIP gap percentage")
