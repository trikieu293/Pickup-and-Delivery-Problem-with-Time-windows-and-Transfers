"""
Cortes 2010 model formulation for PDPTWT.

This module implements the mathematical formulation from:
Cortés, C. E., Matamala, M., & Contardo, C. (2010). 
"The pickup and delivery problem with transfers: Formulation and a branch-and-cut solution method."
European Journal of Operational Research, 200(3), 711-724.

Key features:
- Transfer nodes are split into 'start' (ts) and 'finish' (tf) nodes
- Uses z variables to track request flow
- Includes time window constraints with BigM
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from loguru import logger

from .base import PDPTWTModel
from ..data.schemas import ProblemInstance
from ..solvers.base import BaseSolver, VarType, ObjectiveSense
from ..data.preprocessor import group_nodes, preprocess_instance


class CortesModel(PDPTWTModel):
    """
    Cortes 2010 formulation for PDPTWT.
    
    This formulation splits transfer nodes into start (ts) and finish (tf) nodes
    to model the transfer operation explicitly.
    """
    
    def get_name(self) -> str:
        """Get model name."""
        return "Cortes2010"
    
    def build_variables(self, solver: BaseSolver, instance: ProblemInstance) -> Dict[str, Any]:
        """
        Build decision variables for Cortes model.
        
        Variables:
        - x[k,i,j]: Binary, 1 if vehicle k travels from node i to j
        - z[k,r,i]: Binary, 1 if request r is in vehicle k when leaving node i
        - a[i]: Integer, arrival time at node i
        - ats[k,i]: Integer, arrival time of vehicle k at transfer start node i
        - atf[k,i]: Integer, arrival time of vehicle k at transfer finish node i
        """
        logger.info("Building Cortes model variables...")
        
        # Preprocess instance
        if instance.node_groups is None or instance.distance_matrix is None:
            preprocess_instance(instance)
        
        node_groups = instance.node_groups
        nr = instance.metadata.nr
        nv = instance.metadata.nv
        
        # Expand transfer nodes into ts and tf
        transfer_start = []
        transfer_finish = []
        for t in node_groups.transfer_nodes:
            if not t.endswith('s') and not t.endswith('f'):
                transfer_start.append(f"{t}s")
                transfer_finish.append(f"{t}f")
            elif t.endswith('s'):
                transfer_start.append(t)
            elif t.endswith('f'):
                transfer_finish.append(t)
        
        # Generate arcs for Cortes formulation
        arcs = self._generate_arcs(node_groups, transfer_start, transfer_finish)
        
        logger.info(f"Generated {len(arcs)} arcs")
        
        # Build variable indices
        x_indices = [(k, i, j) for k in range(nv) for (i, j) in arcs]
        z_indices = [(k, r, i) for k in range(nv) for r in range(nr) 
                     for i in node_groups.all_nodes]
        a_indices = list(node_groups.all_nodes)
        ats_indices = [(k, i) for k in range(nv) for i in transfer_start]
        atf_indices = [(k, i) for k in range(nv) for i in transfer_finish]
        
        # Create variables
        x = solver.add_variables(x_indices, 'x', VarType.BINARY)
        z = solver.add_variables(z_indices, 'z', VarType.BINARY)
        a = solver.add_variables(a_indices, 'a', VarType.INTEGER, lb=0)
        ats = solver.add_variables(ats_indices, 'ats', VarType.INTEGER, lb=0)
        atf = solver.add_variables(atf_indices, 'atf', VarType.INTEGER, lb=0)
        
        logger.success(f"Created {len(x_indices)} x variables, {len(z_indices)} z variables")
        
        return {
            'x': x,
            'z': z,
            'a': a,
            'ats': ats,
            'atf': atf,
            'arcs': arcs,
            'transfer_start': transfer_start,
            'transfer_finish': transfer_finish
        }
    
    def _generate_arcs(self, node_groups, transfer_start, transfer_finish):
        """Generate valid arcs for Cortes formulation."""
        arcs = []
        
        # Vehicle origins to pickup nodes, transfer starts, and own destination
        for vo in node_groups.vehicle_origins:
            for p in node_groups.pickup_nodes:
                arcs.append((vo, p))
            for ts in transfer_start:
                arcs.append((vo, ts))
            vd = vo.replace('o', 'e')
            if vd in node_groups.vehicle_destinations:
                arcs.append((vo, vd))
        
        # Pickup nodes to other nodes
        for p in node_groups.pickup_nodes:
            for p2 in node_groups.pickup_nodes:
                if p != p2:
                    arcs.append((p, p2))
            for d in node_groups.delivery_nodes:
                arcs.append((p, d))
            for ts in transfer_start:
                arcs.append((p, ts))
        
        # Delivery nodes to other nodes
        for d in node_groups.delivery_nodes:
            for p in node_groups.pickup_nodes:
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
        for i, ts in enumerate(transfer_start):
            if i < len(transfer_finish):
                arcs.append((ts, transfer_finish[i]))
        
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
                tf_base = tf.replace('tf', '').replace('f', '')
                ts_base = ts.replace('ts', '').replace('s', '')
                if tf_base != ts_base:
                    arcs.append((tf, ts))
        
        return list(set(arcs))
    
    def build_objective(self, solver: BaseSolver, instance: ProblemInstance, variables: Dict[str, Any]) -> None:
        """
        Build objective function: minimize total distance traveled.
        """
        x = variables['x']
        arcs = variables['arcs']
        c = instance.distance_matrix.distances
        nv = instance.metadata.nv
        
        # Get Gurobi model for direct expression building
        from ..solvers.gurobi_solver import GurobiSolver
        if isinstance(solver, GurobiSolver):
            import gurobipy as gp
            model = solver.get_model()
            
            obj_expr = gp.quicksum(
                c[i][j] * x[k, i, j]
                for k in range(nv)
                for (i, j) in arcs
            )
            
            solver.set_objective(obj_expr, ObjectiveSense.MINIMIZE)
            logger.info("Set objective: minimize total distance")
    
    def build_constraints(self, solver: BaseSolver, instance: ProblemInstance, variables: Dict[str, Any]) -> None:
        """Build all constraints for Cortes model."""
        logger.info("Building Cortes model constraints...")
        
        x = variables['x']
        z = variables['z']
        a = variables['a']
        ats = variables['ats']
        atf = variables['atf']
        arcs = variables['arcs']
        transfer_start = variables['transfer_start']
        transfer_finish = variables['transfer_finish']
        
        node_groups = instance.node_groups
        nr = instance.metadata.nr
        nv = instance.metadata.nv
        c = instance.distance_matrix.distances
        
        # Get Gurobi model
        from ..solvers.gurobi_solver import GurobiSolver
        if not isinstance(solver, GurobiSolver):
            raise NotImplementedError("Cortes model currently only supports Gurobi solver")
        
        import gurobipy as gp
        model = solver.get_model()
        
        # Constraint 1: Each vehicle leaves its origin exactly once
        for k in range(nv):
            vo = f"o{k}"
            if vo in node_groups.vehicle_origins:
                model.addConstr(
                    gp.quicksum(x[k, vo, j] for (i, j) in arcs if i == vo) == 1,
                    name=f'vehicle_origin_{k}'
                )
        
        # Constraint 2: Each vehicle arrives at its destination exactly once
        for k in range(nv):
            vd = f"e{k}"
            if vd in node_groups.vehicle_destinations:
                model.addConstr(
                    gp.quicksum(x[k, i, vd] for (i, j) in arcs if j == vd) == 1,
                    name=f'vehicle_destination_{k}'
                )
        
        # Constraint 3: Flow conservation at intermediate nodes
        for k in range(nv):
            for i in list(node_groups.pickup_nodes) + list(node_groups.delivery_nodes):
                model.addConstr(
                    gp.quicksum(x[k, i, j] for (ii, j) in arcs if ii == i) ==
                    gp.quicksum(x[k, j, i] for (j, ii) in arcs if ii == i),
                    name=f'flow_conservation_{k}_{i}'
                )
        
        # Constraint 7: Each delivery node visited exactly once
        for d in node_groups.delivery_nodes:
            model.addConstr(
                gp.quicksum(
                    gp.quicksum(x[k, j, d] for (j, dd) in arcs if dd == d)
                    for k in range(nv)
                ) == 1,
                name=f'delivery_once_{d}'
            )
        
        # Constraint 8: Each pickup node visited exactly once
        for p in node_groups.pickup_nodes:
            model.addConstr(
                gp.quicksum(
                    gp.quicksum(x[k, p, j] for (pp, j) in arcs if pp == p)
                    for k in range(nv)
                ) == 1,
                name=f'pickup_once_{p}'
            )
        
        # Time window constraints
        self._build_time_constraints(model, x, a, ats, atf, arcs, transfer_start, 
                                     transfer_finish, node_groups, instance, nv, nr)
        
        # Request flow constraints
        self._build_request_flow_constraints(model, x, z, arcs, node_groups, nv, nr)
        
        # Capacity constraints
        self._build_capacity_constraints(model, z, node_groups, instance, nv, nr)
        
        logger.success("Cortes model constraints built successfully")
    
    def _build_time_constraints(self, model, x, a, ats, atf, arcs, transfer_start,
                                transfer_finish, node_groups, instance, nv, nr):
        """Build time window constraints."""
        import gurobipy as gp
        c = instance.distance_matrix.distances
        big_m = self.big_m
        
        # Time window constraints for pickup/delivery nodes
        for i in list(node_groups.pickup_nodes) + list(node_groups.delivery_nodes):
            node = instance.get_node(i)
            model.addConstr(a[i] >= node.a, name=f'tw_lower_{i}')
            model.addConstr(a[i] <= node.b, name=f'tw_upper_{i}')
        
        # Time propagation constraints
        for k in range(nv):
            vo = f"o{k}"
            # From vehicle origin
            for (i, j) in arcs:
                if i == vo and j.startswith('p'):
                    model.addConstr(
                        c[i][j] - a[j] <= big_m * (1 - x[k, i, j]),
                        name=f'time_prop_vo_{k}_{j}'
                    )
                if i == vo and j in transfer_start:
                    model.addConstr(
                        c[i][j] - ats[k, j] <= big_m * (1 - x[k, i, j]),
                        name=f'time_prop_vo_ts_{k}_{j}'
                    )
        
        # Between regular nodes
        for (i, j) in arcs:
            if i in list(node_groups.pickup_nodes) + list(node_groups.delivery_nodes):
                if j in list(node_groups.pickup_nodes) + list(node_groups.delivery_nodes):
                    for k in range(nv):
                        model.addConstr(
                            a[i] + c[i][j] - a[j] <= big_m * (1 - x[k, i, j]),
                            name=f'time_prop_{k}_{i}_{j}'
                        )
                elif j in transfer_start:
                    for k in range(nv):
                        model.addConstr(
                            a[i] + c[i][j] - ats[k, j] <= big_m * (1 - x[k, i, j]),
                            name=f'time_prop_ts_{k}_{i}_{j}'
                        )
        
        # Transfer node time constraints
        for i, ts in enumerate(transfer_start):
            if i < len(transfer_finish):
                tf = transfer_finish[i]
                for k in range(nv):
                    model.addConstr(
                        ats[k, ts] + c[ts][tf] - atf[k, tf] <= big_m * (1 - x[k, ts, tf]),
                        name=f'time_transfer_{k}_{ts}_{tf}'
                    )
    
    def _build_request_flow_constraints(self, model, x, z, arcs, node_groups, nv, nr):
        """Build request flow tracking constraints."""
        import gurobipy as gp
        big_m = self.big_m
        
        # Initialize z at vehicle origins/destinations to 0
        for k in range(nv):
            vo = f"o{k}"
            vd = f"e{k}"
            for r in range(nr):
                if vo in node_groups.vehicle_origins:
                    model.addConstr(z[k, r, vo] == 0, name=f'z_init_origin_{k}_{r}')
                if vd in node_groups.vehicle_destinations:
                    model.addConstr(z[k, r, vd] == 0, name=f'z_init_dest_{k}_{r}')
        
        # Request flow propagation
        for k in range(nv):
            for r in range(nr):
                for (i, j) in arcs:
                    pr = f"p{r}"
                    dr = f"d{r}"
                    
                    # Skip transfer arcs
                    if i in node_groups.transfer_nodes and j in node_groups.transfer_nodes:
                        continue
                    
                    # If not pickup or delivery of this request
                    if i != pr and i != dr:
                        model.addConstr(
                            z[k, r, i] - z[k, r, j] <= 1 - x[k, i, j],
                            name=f'request_flow_{k}_{r}_{i}_{j}'
                        )
        
        # Pickup: request enters vehicle
        for k in range(nv):
            for r in range(nr):
                pr = f"p{r}"
                for (i, j) in arcs:
                    if i == pr:
                        model.addConstr(
                            1 - z[k, r, j] <= 1 - x[k, i, j],
                            name=f'pickup_{k}_{r}_{j}'
                        )
        
        # Delivery: request leaves vehicle
        for k in range(nv):
            for r in range(nr):
                dr = f"d{r}"
                for (i, j) in arcs:
                    if i == dr:
                        model.addConstr(
                            z[k, r, j] <= 1 - x[k, i, j],
                            name=f'delivery_{k}_{r}_{j}'
                        )
    
    def _build_capacity_constraints(self, model, z, node_groups, instance, nv, nr):
        """Build vehicle capacity constraints."""
        import gurobipy as gp
        
        capacity = instance.metadata.capacity
        
        # Get load for each request
        loads = {}
        for r in range(nr):
            pr = f"p{r}"
            node = instance.get_node(pr)
            loads[r] = node.load
        
        # Capacity constraint at each node
        for k in range(nv):
            for i in node_groups.all_nodes:
                vo = f"o{k}"
                vd = f"e{k}"
                if i != vo and i != vd:
                    model.addConstr(
                        gp.quicksum(loads[r] * z[k, r, i] for r in range(nr)) <= capacity,
                        name=f'capacity_{k}_{i}'
                    )
