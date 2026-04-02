"""
Rais 2014 model formulation for PDPTWT.

This module implements the mathematical formulation from:
Rais, A., Alvelos, F., & Carvalho, M. S. (2014). 
"New mixed integer-programming model for the pickup-and-delivery problem with transshipment."
European Journal of Operational Research, 235(3), 530-539.

Key features:
- Uses standard transfer nodes (not split like Cortes)
- Variables: x (routing), y (request flow), s (transfer sync), a/b (time windows)
- Simpler arc structure than Cortes
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from loguru import logger

from .base import PDPTWTModel
from ..data.schemas import ProblemInstance
from ..solvers.base import BaseSolver, VarType, ObjectiveSense
from ..data.preprocessor import preprocess_instance


class RaisModel(PDPTWTModel):
    """
    Rais 2014 formulation for PDPTWT.
    
    This formulation uses standard transfer nodes without splitting them.
    """
    
    def get_name(self) -> str:
        """Get model name."""
        return "Rais2014"
    
    def build_variables(self, solver: BaseSolver, instance: ProblemInstance) -> Dict[str, Any]:
        """
        Build decision variables for Rais model.
        
        Variables:
        - x[k,i,j]: Binary, 1 if vehicle k travels from node i to j
        - y[k,r,i,j]: Binary, 1 if request r is transported by vehicle k from i to j
        - s[k1,k2,t,r]: Binary, 1 if request r is transferred from vehicle k1 to k2 at transfer node t
        - a[k,i]: Integer, arrival time of vehicle k at node i
        - b[k,i]: Integer, departure time of vehicle k from node i
        """
        logger.info("Building Rais model variables...")
        
        # Preprocess instance
        if instance.node_groups is None or instance.distance_matrix is None:
            preprocess_instance(instance)
        
        node_groups = instance.node_groups
        nr = instance.metadata.nr
        nv = instance.metadata.nv
        
        # Build variable indices (all arcs between different nodes)
        x_indices = [(k, i, j) for k in range(nv) 
                     for i in node_groups.all_nodes 
                     for j in node_groups.all_nodes if i != j]
        
        y_indices = [(k, r, i, j) for k in range(nv) 
                     for r in range(nr)
                     for i in node_groups.all_nodes 
                     for j in node_groups.all_nodes if i != j]
        
        s_indices = [(k1, k2, t, r) for k1 in range(nv) 
                     for k2 in range(nv) 
                     for t in node_groups.transfer_nodes 
                     for r in range(nr) if k1 != k2]
        
        a_indices = [(k, i) for k in range(nv) for i in node_groups.all_nodes]
        b_indices = [(k, i) for k in range(nv) for i in node_groups.all_nodes]
        
        # Create variables
        x = solver.add_variables(x_indices, 'x', VarType.BINARY)
        y = solver.add_variables(y_indices, 'y', VarType.BINARY)
        s = solver.add_variables(s_indices, 's', VarType.BINARY)
        a = solver.add_variables(a_indices, 'a', VarType.INTEGER, lb=0)
        b = solver.add_variables(b_indices, 'b', VarType.INTEGER, lb=0)
        
        logger.success(f"Created {len(x_indices)} x variables, {len(y_indices)} y variables, {len(s_indices)} s variables")
        
        return {
            'x': x,
            'y': y,
            's': s,
            'a': a,
            'b': b
        }
    
    def build_objective(self, solver: BaseSolver, instance: ProblemInstance, variables: Dict[str, Any]) -> None:
        """
        Build objective function: minimize total distance traveled.
        """
        x = variables['x']
        c = instance.distance_matrix.distances
        nv = instance.metadata.nv
        node_groups = instance.node_groups
        
        # Get Gurobi model for direct expression building
        from ..solvers.gurobi_solver import GurobiSolver
        if isinstance(solver, GurobiSolver):
            import gurobipy as gp
            model = solver.get_model()
            
            obj_expr = gp.quicksum(
                c[i][j] * x[k, i, j]
                for k in range(nv)
                for i in node_groups.all_nodes
                for j in node_groups.all_nodes
                if i != j
            )
            
            solver.set_objective(obj_expr, ObjectiveSense.MINIMIZE)
            logger.info("Set objective: minimize total distance")
    
    def build_constraints(self, solver: BaseSolver, instance: ProblemInstance, variables: Dict[str, Any]) -> None:
        """Build all constraints for Rais model."""
        logger.info("Building Rais model constraints...")
        
        x = variables['x']
        y = variables['y']
        s = variables['s']
        a = variables['a']
        b = variables['b']
        
        node_groups = instance.node_groups
        nr = instance.metadata.nr
        nv = instance.metadata.nv
        c = instance.distance_matrix.distances
        
        # Get Gurobi model
        from ..solvers.gurobi_solver import GurobiSolver
        if not isinstance(solver, GurobiSolver):
            raise NotImplementedError("Rais model currently only supports Gurobi solver")
        
        import gurobipy as gp
        model = solver.get_model()
        
        # Constraint 1: Each vehicle leaves its origin exactly once
        for k in range(nv):
            vo = f"o{k}"
            if vo in node_groups.vehicle_origins:
                model.addConstr(
                    gp.quicksum(x[k, vo, j] for j in node_groups.all_nodes if vo != j) == 1,
                    name=f'vehicle_origin_{k}'
                )
        
        # Constraint 2: Vehicle flow balance (origin to destination)
        for k in range(nv):
            vo = f"o{k}"
            vd = f"e{k}"
            if vo in node_groups.vehicle_origins and vd in node_groups.vehicle_destinations:
                model.addConstr(
                    gp.quicksum(x[k, vo, j] for j in node_groups.all_nodes if vo != j) ==
                    gp.quicksum(x[k, j, vd] for j in node_groups.all_nodes if vd != j),
                    name=f'vehicle_balance_{k}'
                )
        
        # Constraint 3: Flow conservation at intermediate nodes
        for k in range(nv):
            vo = f"o{k}"
            vd = f"e{k}"
            for i in node_groups.all_nodes:
                if i != vo and i != vd:
                    model.addConstr(
                        gp.quicksum(x[k, i, j] for j in node_groups.all_nodes if i != j) ==
                        gp.quicksum(x[k, j, i] for j in node_groups.all_nodes if i != j),
                        name=f'flow_conservation_{k}_{i}'
                    )
        
        # Request flow constraints
        self._build_request_flow_constraints(model, x, y, s, node_groups, nr, nv)
        
        # Capacity constraints
        self._build_capacity_constraints(model, x, y, instance, node_groups, nr, nv)
        
        # Time window constraints
        self._build_time_constraints(model, x, a, b, c, instance, node_groups, nv)
        
        # Transfer synchronization constraints
        self._build_transfer_constraints(model, y, s, a, b, instance, node_groups, nr, nv)
        
        logger.success("Rais model constraints built successfully")
    
    def _build_request_flow_constraints(self, model, x, y, s, node_groups, nr, nv):
        """Build request flow constraints."""
        import gurobipy as gp
        
        # Constraint 4: Each request picked up exactly once
        for r in range(nr):
            pr = f"p{r}"
            if pr in node_groups.pickup_nodes:
                model.addConstr(
                    gp.quicksum(
                        gp.quicksum(y[k, r, pr, j] for j in node_groups.all_nodes if pr != j)
                        for k in range(nv)
                    ) == 1,
                    name=f'pickup_once_{r}'
                )
        
        # Constraint 5: Each request delivered exactly once
        for r in range(nr):
            dr = f"d{r}"
            if dr in node_groups.delivery_nodes:
                model.addConstr(
                    gp.quicksum(
                        gp.quicksum(y[k, r, j, dr] for j in node_groups.all_nodes if dr != j)
                        for k in range(nv)
                    ) == 1,
                    name=f'delivery_once_{r}'
                )
        
        # Constraint 6: Flow balance at transfer nodes
        for r in range(nr):
            for t in node_groups.transfer_nodes:
                model.addConstr(
                    gp.quicksum(
                        gp.quicksum(y[k, r, t, j] for j in node_groups.all_nodes if t != j)
                        for k in range(nv)
                    ) ==
                    gp.quicksum(
                        gp.quicksum(y[k, r, j, t] for j in node_groups.all_nodes if t != j)
                        for k in range(nv)
                    ),
                    name=f'transfer_balance_{r}_{t}'
                )
        
        # Constraint 16: Flow conservation at non-pickup/delivery/transfer nodes
        for r in range(nr):
            pr = f"p{r}"
            dr = f"d{r}"
            for k in range(nv):
                for i in node_groups.all_nodes:
                    if (i != pr and i != dr and 
                        i not in node_groups.transfer_nodes):
                        model.addConstr(
                            gp.quicksum(y[k, r, i, j] for j in node_groups.all_nodes if i != j) ==
                            gp.quicksum(y[k, r, j, i] for j in node_groups.all_nodes if i != j),
                            name=f'request_flow_{k}_{r}_{i}'
                        )
        
        # Constraint 8: Request flow follows vehicle flow
        for k in range(nv):
            for r in range(nr):
                for i in node_groups.all_nodes:
                    for j in node_groups.all_nodes:
                        if i != j:
                            model.addConstr(
                                y[k, r, i, j] <= x[k, i, j],
                                name=f'flow_sync_{k}_{r}_{i}_{j}'
                            )
    
    def _build_capacity_constraints(self, model, x, y, instance, node_groups, nr, nv):
        """Build vehicle capacity constraints."""
        import gurobipy as gp
        
        capacity = instance.metadata.capacity
        
        # Get load for each request
        loads = {}
        for r in range(nr):
            pr = f"p{r}"
            node = instance.get_node(pr)
            loads[r] = node.load
        
        # Constraint 9: Capacity constraint on each arc
        for k in range(nv):
            for i in node_groups.all_nodes:
                for j in node_groups.all_nodes:
                    if i != j:
                        model.addConstr(
                            gp.quicksum(loads[r] * y[k, r, i, j] for r in range(nr)) <= 
                            capacity * x[k, i, j],
                            name=f'capacity_{k}_{i}_{j}'
                        )
    
    def _build_time_constraints(self, model, x, a, b, c, instance, node_groups, nv):
        """Build time window constraints."""
        import gurobipy as gp
        
        # Constraint 15: Time propagation
        for k in range(nv):
            for i in node_groups.all_nodes:
                for j in node_groups.all_nodes:
                    if i != j:
                        node_i = instance.get_node(i)
                        node_j = instance.get_node(j)
                        M = max(0, node_i.b) + c[i][j] - node_j.a
                        
                        model.addConstr(
                            b[k, i] + c[i][j] - a[k, j] <= M * (1 - x[k, i, j]),
                            name=f'time_prop_{k}_{i}_{j}'
                        )
        
        # Constraint 16: Arrival before departure
        for k in range(nv):
            for i in node_groups.all_nodes:
                model.addConstr(
                    a[k, i] <= b[k, i],
                    name=f'arrival_departure_{k}_{i}'
                )
        
        # Constraints 17-18: Time windows for pickup and delivery nodes
        for k in range(nv):
            for i in list(node_groups.pickup_nodes) + list(node_groups.delivery_nodes):
                node = instance.get_node(i)
                model.addConstr(
                    a[k, i] >= node.a,
                    name=f'tw_lower_{k}_{i}'
                )
                model.addConstr(
                    b[k, i] <= node.b,
                    name=f'tw_upper_{k}_{i}'
                )
    
    def _build_transfer_constraints(self, model, y, s, a, b, instance, node_groups, nr, nv):
        """Build transfer synchronization constraints."""
        import gurobipy as gp
        
        # Constraint 19: Transfer synchronization variable
        for r in range(nr):
            for t in node_groups.transfer_nodes:
                node_t = instance.get_node(t)
                M = node_t.b - node_t.a
                
                for k1 in range(nv):
                    for k2 in range(nv):
                        if k1 != k2:
                            # Link s variable to y variables
                            model.addConstr(
                                gp.quicksum(y[k1, r, j, t] for j in node_groups.all_nodes if j != t) +
                                gp.quicksum(y[k2, r, t, j] for j in node_groups.all_nodes if j != t) <=
                                s[k1, k2, t, r] + 1,
                                name=f'transfer_sync_{k1}_{k2}_{t}_{r}'
                            )
                            
                            # Constraint 20: Time synchronization at transfer
                            model.addConstr(
                                a[k1, t] - b[k2, t] <= M * (1 - s[k1, k2, t, r]),
                                name=f'transfer_time_{k1}_{k2}_{t}_{r}'
                            )
