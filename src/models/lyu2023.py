"""
Lyu 2023 model formulation for PDPTWT.

This module implements the mathematical formulation from:
Lyu, Z., & Yu, V. F. (2023). 
"An exact algorithm for the pickup and delivery problem with time windows and transfers."
Computers & Operations Research, 150, 106080.

Key features:
- Enhanced formulation with additional valid inequalities
- Variables: x (routing), y (request flow), s (transfer sync), a/b (time windows)
- Includes constraints 40-47 as valid inequalities
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from loguru import logger

from .base import PDPTWTModel
from ..data.schemas import ProblemInstance
from ..solvers.base import BaseSolver, VarType, ObjectiveSense
from ..data.preprocessor import preprocess_instance


class LyuModel(PDPTWTModel):
    """
    Lyu 2023 formulation for PDPTWT.
    
    This formulation includes additional valid inequalities to strengthen
    the linear relaxation.
    """
    
    def get_name(self) -> str:
        """Get model name."""
        return "Lyu2023"
    
    def build_variables(self, solver: BaseSolver, instance: ProblemInstance) -> Dict[str, Any]:
        """
        Build decision variables for Lyu model.
        
        Variables:
        - x[k,i,j]: Binary, 1 if vehicle k travels from node i to j
        - y[k,r,i,j]: Binary, 1 if request r is transported by vehicle k from i to j
        - s[k1,k2,t,r]: Binary, 1 if request r is transferred from vehicle k1 to k2 at transfer node t
        - a[k,i]: Integer, arrival time of vehicle k at node i
        - b[k,i]: Integer, departure time of vehicle k from node i
        """
        logger.info("Building Lyu model variables...")
        
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
        """Build all constraints for Lyu model."""
        logger.info("Building Lyu model constraints...")
        
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
            raise NotImplementedError("Lyu model currently only supports Gurobi solver")
        
        import gurobipy as gp
        model = solver.get_model()
        
        # Constraints 25, 27, 42: Vehicle flow
        self._build_vehicle_flow_constraints(model, x, node_groups, nv)
        
        # Constraints 4, 5, 6, 16: Request flow
        self._build_request_flow_constraints(model, x, y, node_groups, nr, nv)
        
        # Constraints 8, 9: Synchronization and capacity
        self._build_sync_capacity_constraints(model, x, y, instance, node_groups, nr, nv)
        
        # Constraints 21, 48: Transfer synchronization
        self._build_transfer_constraints(model, y, s, a, b, instance, node_groups, nr, nv)
        
        # Constraints 49-51: Time windows
        self._build_time_constraints(model, x, a, b, c, instance, node_groups, nv)
        
        # Constraints 40-47: Valid inequalities
        self._build_valid_inequalities(model, x, y, node_groups, nr, nv)
        
        logger.success("Lyu model constraints built successfully")
    
    def _build_vehicle_flow_constraints(self, model, x, node_groups, nv):
        """Build vehicle flow constraints (25, 27, 42)."""
        import gurobipy as gp
        
        # Constraint 25: Each vehicle leaves its origin exactly once
        for k in range(nv):
            vo = f"o{k}"
            if vo in node_groups.vehicle_origins:
                model.addConstr(
                    gp.quicksum(x[k, vo, j] for j in node_groups.all_nodes if vo != j) == 1,
                    name=f'constr25_{k}'
                )
        
        # Constraint 42: Each vehicle arrives at its destination exactly once
        for k in range(nv):
            vd = f"e{k}"
            if vd in node_groups.vehicle_destinations:
                model.addConstr(
                    gp.quicksum(x[k, j, vd] for j in node_groups.all_nodes if vd != j) == 1,
                    name=f'constr42_{k}'
                )
        
        # Constraint 27: Flow conservation at intermediate nodes
        for k in range(nv):
            for i in list(node_groups.pickup_nodes) + list(node_groups.delivery_nodes) + list(node_groups.transfer_nodes):
                model.addConstr(
                    gp.quicksum(x[k, i, j] for j in node_groups.all_nodes if i != j) -
                    gp.quicksum(x[k, j, i] for j in node_groups.all_nodes if i != j) == 0,
                    name=f'constr27_{k}_{i}'
                )
    
    def _build_request_flow_constraints(self, model, x, y, node_groups, nr, nv):
        """Build request flow constraints (4, 5, 6, 16)."""
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
                    name=f'constr4_{r}'
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
                    name=f'constr5_{r}'
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
                    name=f'constr6_{r}_{t}'
                )
        
        # Constraint 16: Flow conservation at non-special nodes
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
                            name=f'constr16_{k}_{r}_{i}'
                        )
    
    def _build_sync_capacity_constraints(self, model, x, y, instance, node_groups, nr, nv):
        """Build synchronization and capacity constraints (8, 9)."""
        import gurobipy as gp
        
        capacity = instance.metadata.capacity
        
        # Get load for each request
        loads = {}
        for r in range(nr):
            pr = f"p{r}"
            node = instance.get_node(pr)
            loads[r] = node.load
        
        # Constraints 8 and 9
        for k in range(nv):
            for i in node_groups.all_nodes:
                for j in node_groups.all_nodes:
                    if i != j:
                        # Constraint 8: Request flow follows vehicle flow
                        for r in range(nr):
                            model.addConstr(
                                y[k, r, i, j] <= x[k, i, j],
                                name=f'constr8_{k}_{r}_{i}_{j}'
                            )
                        
                        # Constraint 9: Capacity constraint
                        model.addConstr(
                            gp.quicksum(loads[r] * y[k, r, i, j] for r in range(nr)) <= 
                            capacity * x[k, i, j],
                            name=f'constr9_{k}_{i}_{j}'
                        )
    
    def _build_transfer_constraints(self, model, y, s, a, b, instance, node_groups, nr, nv):
        """Build transfer synchronization constraints (21, 48)."""
        import gurobipy as gp
        
        # Constraints 21 and 48
        for r in range(nr):
            for t in node_groups.transfer_nodes:
                node_t = instance.get_node(t)
                M = node_t.b - node_t.a
                
                for k1 in range(nv):
                    for k2 in range(nv):
                        if k1 != k2:
                            # Constraint 21: Transfer synchronization variable
                            model.addConstr(
                                gp.quicksum(y[k1, r, j, t] for j in node_groups.all_nodes if j != t) +
                                gp.quicksum(y[k2, r, t, j] for j in node_groups.all_nodes if j != t) <=
                                s[k1, k2, t, r] + 1,
                                name=f'constr21_{k1}_{k2}_{t}_{r}'
                            )
                            
                            # Constraint 48: Time synchronization at transfer
                            model.addConstr(
                                a[k1, t] - b[k2, t] <= M * (1 - s[k1, k2, t, r]),
                                name=f'constr48_{k1}_{k2}_{t}_{r}'
                            )
    
    def _build_time_constraints(self, model, x, a, b, c, instance, node_groups, nv):
        """Build time window constraints (49-51)."""
        import gurobipy as gp
        
        # Constraint 49: Time propagation
        for k in range(nv):
            for i in node_groups.all_nodes:
                for j in node_groups.all_nodes:
                    if i != j:
                        node_i = instance.get_node(i)
                        node_j = instance.get_node(j)
                        M = max(0, node_i.b) + c[i][j] - node_j.a
                        
                        model.addConstr(
                            b[k, i] + c[i][j] - a[k, j] <= M * (1 - x[k, i, j]),
                            name=f'constr49_{k}_{i}_{j}'
                        )
        
        # Constraints 50 and 51: Time windows
        for k in range(nv):
            for i in node_groups.all_nodes:
                node = instance.get_node(i)
                
                # Constraint 50a: Lower bound
                model.addConstr(
                    a[k, i] >= node.a,
                    name=f'constr50a_{k}_{i}'
                )
                
                # Constraint 50b: Upper bound
                model.addConstr(
                    b[k, i] <= node.b,
                    name=f'constr50b_{k}_{i}'
                )
                
                # Constraint 51: Arrival before departure
                model.addConstr(
                    a[k, i] <= b[k, i],
                    name=f'constr51_{k}_{i}'
                )
    
    def _build_valid_inequalities(self, model, x, y, node_groups, nr, nv):
        """Build valid inequalities (40-47)."""
        import gurobipy as gp
        
        # Constraint 40: Vehicles cannot return to origin
        for k in range(nv):
            vo = f"o{k}"
            if vo in node_groups.vehicle_origins:
                model.addConstr(
                    gp.quicksum(x[k, j, vo] for j in node_groups.all_nodes if j != vo) == 0,
                    name=f'constr40_{k}'
                )
        
        # Constraint 41: Vehicles cannot depart from other origins
        for k in range(nv):
            for i in list(node_groups.vehicle_origins) + list(node_groups.vehicle_destinations):
                vo = f"o{k}"
                if i != vo:
                    model.addConstr(
                        gp.quicksum(x[k, i, j] for j in node_groups.all_nodes if i != j) == 0,
                        name=f'constr41_{k}_{i}'
                    )
        
        # Constraint 43: Vehicles cannot leave their destination
        for k in range(nv):
            vd = f"e{k}"
            if vd in node_groups.vehicle_destinations:
                model.addConstr(
                    gp.quicksum(x[k, vd, j] for j in node_groups.all_nodes if vd != j) == 0,
                    name=f'constr43_{k}'
                )
        
        # Constraint 44: Each vehicle visits transfer station at most once
        for k in range(nv):
            for t in node_groups.transfer_nodes:
                model.addConstr(
                    gp.quicksum(x[k, t, j] for j in node_groups.all_nodes if t != j) <= 1,
                    name=f'constr44_{k}_{t}'
                )
        
        # Constraint 45: Pickup and delivery locations visited exactly once
        for i in list(node_groups.pickup_nodes) + list(node_groups.delivery_nodes):
            model.addConstr(
                gp.quicksum(
                    gp.quicksum(x[k, i, j] for j in node_groups.all_nodes if i != j)
                    for k in range(nv)
                ) == 1,
                name=f'constr45_{i}'
            )
        
        # Constraint 46: No arcs heading to pickup locations in request flow
        for r in range(nr):
            pr = f"p{r}"
            if pr in node_groups.pickup_nodes:
                model.addConstr(
                    gp.quicksum(
                        gp.quicksum(y[k, r, i, pr] for i in node_groups.all_nodes if i != pr)
                        for k in range(nv)
                    ) == 0,
                    name=f'constr46_{r}'
                )
        
        # Constraint 47: Request flows exclude vehicle depots
        for r in range(nr):
            for k in range(nv):
                vo = f"o{k}"
                vd = f"e{k}"
                for i in list(node_groups.vehicle_origins) + list(node_groups.vehicle_destinations):
                    if i != vo and i != vd:
                        model.addConstr(
                            gp.quicksum(y[k, r, i, j] for j in node_groups.all_nodes if i != j) == 0,
                            name=f'constr47_{k}_{r}_{i}'
                        )
