import networkx as nx
import math
import random
from typing import Dict, List

class SingleDstPath(Dict):
    partition: int
    edges: List[List]

class BroadCastTopology:
    def __init__(self, src: str, dsts: List[str], num_partitions: int = 4, paths: Dict[str, SingleDstPath] = None):
        self.src = src
        self.dsts = dsts
        self.num_partitions = num_partitions
        if paths is not None:
            self.paths = paths
            self.set_graph()
        else:
            self.paths = {dst: {str(i): None for i in range(num_partitions)} for dst in dsts}

    def get_paths(self):
        return self.paths

    def set_num_partitions(self, num_partitions: int):
        self.num_partitions = num_partitions

    def set_dst_partition_paths(self, dst: str, partition: int, paths: List[List]):
        partition = str(partition)
        self.paths[dst][partition] = paths

    def append_dst_partition_path(self, dst: str, partition: int, path: List):
        partition = str(partition)
        if self.paths[dst][partition] is None:
            self.paths[dst][partition] = []
        self.paths[dst][partition].append(path)

def search_algorithm(src, dsts, G, num_partitions):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))

    nodes = list(h.nodes())
    edges = list(h.edges(data=True))
    
    # Precompute edge dictionary for SPH
    edges_dict = {u: {} for u in nodes}
    for u, v, data in edges:
        edges_dict[u][v] = data

    def evaluate_edges(selected_edges):
        sub = nx.DiGraph()
        sub.add_nodes_from(nodes)
        sub.add_edges_from(selected_edges)
        
        tree_edges = set()
        for dst in dsts:
            try:
                def weight_func(u, v, d):
                    c_val = h[u][v].get('cost')
                    return 1000.0 if c_val is None else float(c_val)
                path = nx.dijkstra_path(sub, src, dst, weight=weight_func)
                for i in range(len(path) - 1):
                    tree_edges.add((path[i], path[i+1]))
            except nx.NetworkXNoPath:
                return float('inf'), None
                
        total_cost = sum(
            (1000.0 if h[u][v].get('cost') is None else float(h[u][v].get('cost')))
            for u, v in tree_edges
        )
        return total_cost, list(tree_edges)

    best_cost = float('inf')
    best_tree = None

    def update_best(c, t):
        nonlocal best_cost, best_tree
        if c < best_cost:
            best_cost = c
            best_tree = t

    # Base SPH Approximation (Nearest first)
    def get_sph_tree(order=None):
        import heapq
        T_nodes = {src}
        T_edges = set()
        
        current_cost = {u: {} for u in nodes}
        for u, v_dict in edges_dict.items():
            for v, data in v_dict.items():
                current_cost[u][v] = 1000.0 if data.get('cost') is None else float(data['cost'])
                
        unreached = set(dsts) if order is None else list(order)
        
        while unreached:
            dist = {n: float('inf') for n in nodes}
            prev = {n: None for n in nodes}
            pq = []
            for n in T_nodes:
                dist[n] = 0
                heapq.heappush(pq, (0, n))
                
            found_dst = None
            
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue
                    
                # If no specific order, pick the closest unreached destination
                if order is None and u in unreached:
                    found_dst = u
                    break
                # If specific order, only break when we reach the NEXT destination in the list
                elif order is not None and u == unreached[0]:
                    found_dst = u
                    break
                    
                for v, c in current_cost[u].items():
                    alt = d + c
                    if alt < dist[v]:
                        dist[v] = alt
                        prev[v] = u
                        heapq.heappush(pq, (alt, v))
                        
            if found_dst is None:
                return None
                
            path = []
            curr = found_dst
            while curr not in T_nodes:
                p = prev[curr]
                path.append((p, curr))
                curr = p
            path.reverse()
            
            for u, v in path:
                T_nodes.add(v)
                T_edges.add((u, v))
                current_cost[u][v] = 0.0
                
            if order is None:
                unreached.remove(found_dst)
            else:
                unreached.pop(0)
                
        return list(T_edges)

    # 1. Evaluate standard SPH
    t_sph = get_sph_tree(order=None)
    if t_sph is not None:
        c_sph, t_sph_eval = evaluate_edges(t_sph)
        update_best(c_sph, t_sph_eval)

    # 2. Evaluate randomized SPH orderings
    for _ in range(20):
        shuffled_dsts = list(dsts)
        random.shuffle(shuffled_dsts)
        t_rand_sph = get_sph_tree(order=shuffled_dsts)
        if t_rand_sph is not None:
            c_rand, t_rand_eval = evaluate_edges(t_rand_sph)
            update_best(c_rand, t_rand_eval)

    # 3. Fractional Multi-Commodity Flow via Continuous LP Relaxation
    try:
        import scipy.sparse as sp
        from scipy.optimize import linprog
        
        num_e = len(edges)
        num_d = len(dsts)
        num_vars = num_e + num_d * num_e

        c = [0.0] * num_vars
        for j, (u, v, data) in enumerate(edges):
            cost = data.get('cost')
            c[j] = 1000.0 if cost is None else float(cost)

        A_eq_rows = []
        A_eq_cols = []
        A_eq_data = []
        b_eq = []
        
        row_idx = 0
        for d_idx, d in enumerate(dsts):
            for v in nodes:
                if v == src:
                    continue
                b_val = 1 if v == d else 0
                b_eq.append(b_val)
                for j, (u_edge, v_edge, _) in enumerate(edges):
                    if v_edge == v: 
                        A_eq_rows.append(row_idx)
                        A_eq_cols.append(num_e + d_idx * num_e + j)
                        A_eq_data.append(1.0)
                    if u_edge == v:
                        A_eq_rows.append(row_idx)
                        A_eq_cols.append(num_e + d_idx * num_e + j)
                        A_eq_data.append(-1.0)
                row_idx += 1
                
        if row_idx > 0:
            A_eq = sp.csr_matrix((A_eq_data, (A_eq_rows, A_eq_cols)), shape=(row_idx, num_vars))
        else:
            A_eq = None
            b_eq = None

        A_ub_rows = []
        A_ub_cols = []
        A_ub_data = []
        b_ub = [0.0] * (num_d * num_e)
        
        row_idx_ub = 0
        for d_idx in range(num_d):
            for j in range(num_e):
                A_ub_rows.append(row_idx_ub)
                A_ub_cols.append(j)
                A_ub_data.append(-1.0)
                A_ub_rows.append(row_idx_ub)
                A_ub_cols.append(num_e + d_idx * num_e + j)
                A_ub_data.append(1.0)
                row_idx_ub += 1
                
        if row_idx_ub > 0:
            A_ub = sp.csr_matrix((A_ub_data, (A_ub_rows, A_ub_cols)), shape=(row_idx_ub, num_vars))
        else:
            A_ub = None
            b_ub = None

        bounds = [(0, 1) for _ in range(num_vars)]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if res.success:
            x_e = res.x[:num_e]
            
            # Strategy: Log-transformed LP Weighting
            for j, (u, v, _) in enumerate(edges):
                h[u][v]['lp_weight'] = -math.log(max(x_e[j], 1e-5))
                
            tree_edges_lp = set()
            valid_lp = True
            for dst in dsts:
                try:
                    path = nx.dijkstra_path(h, src, dst, weight='lp_weight')
                    for i in range(len(path) - 1):
                        tree_edges_lp.add((path[i], path[i+1]))
                except nx.NetworkXNoPath:
                    valid_lp = False
                    break
                    
            if valid_lp:
                c_lp, t_lp = evaluate_edges(list(tree_edges_lp))
                update_best(c_lp, t_lp)
                
            # Strategy: Randomized Rounding
            for _ in range(50):
                sampled = []
                for j, (u, v, _) in enumerate(edges):
                    if random.random() < x_e[j]:
                        sampled.append((u, v))
                c_rand, t_rand = evaluate_edges(sampled)
                update_best(c_rand, t_rand)
                
    except Exception as e:
        # Fallback fully handles the case where scipy cannot be imported
        pass

    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    
    if best_tree is None:
        return bc_topology
        
    best_sub = nx.DiGraph()
    best_sub.add_nodes_from(nodes)
    for u, v in best_tree:
        best_sub.add_edge(u, v, **h[u][v])
        
    for dst in dsts:
        try:
            path = nx.dijkstra_path(best_sub, src, dst, weight=lambda u, v, d: 1000.0 if d.get('cost') is None else float(d.get('cost')))
            for i in range(len(path) - 1):
                s_node, t_node = path[i], path[i + 1]
                edge_data = G[s_node][t_node]
                for p in range(num_partitions):
                    bc_topology.append_dst_partition_path(dst, p, [s_node, t_node, edge_data])
        except nx.NetworkXNoPath:
            pass

    return bc_topology

class Solution:
    def solve(self):
        return {"code": """import networkx as nx
import math
import random
from typing import Dict, List

class SingleDstPath(Dict):
    partition: int
    edges: List[List]

class BroadCastTopology:
    def __init__(self, src: str, dsts: List[str], num_partitions: int = 4, paths: Dict[str, SingleDstPath] = None):
        self.src = src
        self.dsts = dsts
        self.num_partitions = num_partitions
        if paths is not None:
            self.paths = paths
            self.set_graph()
        else:
            self.paths = {dst: {str(i): None for i in range(num_partitions)} for dst in dsts}

    def get_paths(self):
        return self.paths

    def set_num_partitions(self, num_partitions: int):
        self.num_partitions = num_partitions

    def set_dst_partition_paths(self, dst: str, partition: int, paths: List[List]):
        partition = str(partition)
        self.paths[dst][partition] = paths

    def append_dst_partition_path(self, dst: str, partition: int, path: List):
        partition = str(partition)
        if self.paths[dst][partition] is None:
            self.paths[dst][partition] = []
        self.paths[dst][partition].append(path)

def search_algorithm(src, dsts, G, num_partitions):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))

    nodes = list(h.nodes())
    edges = list(h.edges(data=True))
    
    edges_dict = {u: {} for u in nodes}
    for u, v, data in edges:
        edges_dict[u][v] = data

    def evaluate_edges(selected_edges):
        sub = nx.DiGraph()
        sub.add_nodes_from(nodes)
        sub.add_edges_from(selected_edges)
        
        tree_edges = set()
        for dst in dsts:
            try:
                def weight_func(u, v, d):
                    c_val = h[u][v].get('cost')
                    return 1000.0 if c_val is None else float(c_val)
                path = nx.dijkstra_path(sub, src, dst, weight=weight_func)
                for i in range(len(path) - 1):
                    tree_edges.add((path[i], path[i+1]))
            except nx.NetworkXNoPath:
                return float('inf'), None
                
        total_cost = sum(
            (1000.0 if h[u][v].get('cost') is None else float(h[u][v].get('cost')))
            for u, v in tree_edges
        )
        return total_cost, list(tree_edges)

    best_cost = float('inf')
    best_tree = None

    def update_best(c, t):
        nonlocal best_cost, best_tree
        if c < best_cost:
            best_cost = c
            best_tree = t

    def get_sph_tree(order=None):
        import heapq
        T_nodes = {src}
        T_edges = set()
        
        current_cost = {u: {} for u in nodes}
        for u, v_dict in edges_dict.items():
            for v, data in v_dict.items():
                current_cost[u][v] = 1000.0 if data.get('cost') is None else float(data['cost'])
                
        unreached = set(dsts) if order is None else list(order)
        
        while unreached:
            dist = {n: float('inf') for n in nodes}
            prev = {n: None for n in nodes}
            pq = []
            for n in T_nodes:
                dist[n] = 0
                heapq.heappush(pq, (0, n))
                
            found_dst = None
            
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue
                    
                if order is None and u in unreached:
                    found_dst = u
                    break
                elif order is not None and u == unreached[0]:
                    found_dst = u
                    break
                    
                for v, c in current_cost[u].items():
                    alt = d + c
                    if alt < dist[v]:
                        dist[v] = alt
                        prev[v] = u
                        heapq.heappush(pq, (alt, v))
                        
            if found_dst is None:
                return None
                
            path = []
            curr = found_dst
            while curr not in T_nodes:
                p = prev[curr]
                path.append((p, curr))
                curr = p
            path.reverse()
            
            for u, v in path:
                T_nodes.add(v)
                T_edges.add((u, v))
                current_cost[u][v] = 0.0
                
            if order is None:
                unreached.remove(found_dst)
            else:
                unreached.pop(0)
                
        return list(T_edges)

    t_sph = get_sph_tree(order=None)
    if t_sph is not None:
        c_sph, t_sph_eval = evaluate_edges(t_sph)
        update_best(c_sph, t_sph_eval)

    for _ in range(20):
        shuffled_dsts = list(dsts)
        random.shuffle(shuffled_dsts)
        t_rand_sph = get_sph_tree(order=shuffled_dsts)
        if t_rand_sph is not None:
            c_rand, t_rand_eval = evaluate_edges(t_rand_sph)
            update_best(c_rand, t_rand_eval)

    try:
        import scipy.sparse as sp
        from scipy.optimize import linprog
        
        num_e = len(edges)
        num_d = len(dsts)
        num_vars = num_e + num_d * num_e

        c = [0.0] * num_vars
        for j, (u, v, data) in enumerate(edges):
            cost = data.get('cost')
            c[j] = 1000.0 if cost is None else float(cost)

        A_eq_rows = []
        A_eq_cols = []
        A_eq_data = []
        b_eq = []
        
        row_idx = 0
        for d_idx, d in enumerate(dsts):
            for v in nodes:
                if v == src:
                    continue
                b_val = 1 if v == d else 0
                b_eq.append(b_val)
                for j, (u_edge, v_edge, _) in enumerate(edges):
                    if v_edge == v: 
                        A_eq_rows.append(row_idx)
                        A_eq_cols.append(num_e + d_idx * num_e + j)
                        A_eq_data.append(1.0)
                    if u_edge == v:
                        A_eq_rows.append(row_idx)
                        A_eq_cols.append(num_e + d_idx * num_e + j)
                        A_eq_data.append(-1.0)
                row_idx += 1
                
        if row_idx > 0:
            A_eq = sp.csr_matrix((A_eq_data, (A_eq_rows, A_eq_cols)), shape=(row_idx, num_vars))
        else:
            A_eq = None
            b_eq = None

        A_ub_rows = []
        A_ub_cols = []
        A_ub_data = []
        b_ub = [0.0] * (num_d * num_e)
        
        row_idx_ub = 0
        for d_idx in range(num_d):
            for j in range(num_e):
                A_ub_rows.append(row_idx_ub)
                A_ub_cols.append(j)
                A_ub_data.append(-1.0)
                A_ub_rows.append(row_idx_ub)
                A_ub_cols.append(num_e + d_idx * num_e + j)
                A_ub_data.append(1.0)
                row_idx_ub += 1
                
        if row_idx_ub > 0:
            A_ub = sp.csr_matrix((A_ub_data, (A_ub_rows, A_ub_cols)), shape=(row_idx_ub, num_vars))
        else:
            A_ub = None
            b_ub = None

        bounds = [(0, 1) for _ in range(num_vars)]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if res.success:
            x_e = res.x[:num_e]
            
            for j, (u, v, _) in enumerate(edges):
                h[u][v]['lp_weight'] = -math.log(max(x_e[j], 1e-5))
                
            tree_edges_lp = set()
            valid_lp = True
            for dst in dsts:
                try:
                    path = nx.dijkstra_path(h, src, dst, weight='lp_weight')
                    for i in range(len(path) - 1):
                        tree_edges_lp.add((path[i], path[i+1]))
                except nx.NetworkXNoPath:
                    valid_lp = False
                    break
                    
            if valid_lp:
                c_lp, t_lp = evaluate_edges(list(tree_edges_lp))
                update_best(c_lp, t_lp)
                
            for _ in range(50):
                sampled = []
                for j, (u, v, _) in enumerate(edges):
                    if random.random() < x_e[j]:
                        sampled.append((u, v))
                c_rand, t_rand = evaluate_edges(sampled)
                update_best(c_rand, t_rand)
                
    except Exception as e:
        pass

    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    
    if best_tree is None:
        return bc_topology
        
    best_sub = nx.DiGraph()
    best_sub.add_nodes_from(nodes)
    for u, v in best_tree:
        best_sub.add_edge(u, v, **h[u][v])
        
    for dst in dsts:
        try:
            path = nx.dijkstra_path(best_sub, src, dst, weight=lambda u, v, d: 1000.0 if d.get('cost') is None else float(d.get('cost')))
            for i in range(len(path) - 1):
                s_node, t_node = path[i], path[i + 1]
                edge_data = G[s_node][t_node]
                for p in range(num_partitions):
                    bc_topology.append_dst_partition_path(dst, p, [s_node, t_node, edge_data])
        except nx.NetworkXNoPath:
            pass

    return bc_topology"""}