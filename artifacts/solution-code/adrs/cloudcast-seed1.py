import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List
import itertools
from itertools import combinations
import random

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

def make_nx_graph(cost_path=None, throughput_path=None, num_vms=1):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if cost_path is None:
        cost = pd.read_csv(os.path.join(current_dir, "profiles/cost.csv"))
    else:
        cost = pd.read_csv(cost_path)
    if throughput_path is None:
        throughput = pd.read_csv(os.path.join(current_dir, "profiles/throughput.csv"))
    else:
        throughput = pd.read_csv(throughput_path)

    G = nx.DiGraph()
    for _, row in throughput.iterrows():
        if row["src_region"] == row["dst_region"]:
            continue
        G.add_edge(row["src_region"], row["dst_region"], cost=None, throughput=num_vms * row["throughput_sent"] / 1e9)

    for _, row in cost.iterrows():
        if row["src"] in G and row["dest"] in G[row["src"]]:
            G[row["src"]][row["dest"]]["cost"] = row["cost"]

    no_cost_pairs = []
    for edge in G.edges.data():
        src_n, dst_n = edge[0], edge[1]
        if edge[-1]["cost"] is None:
            no_cost_pairs.append((src_n, dst_n))
    return G

def prune_leaves(tree_G, src, dsts):
    if tree_G is None:
        return None
    dst_set = set(dsts)
    changed = True
    while changed:
        changed = False
        nodes = list(tree_G.nodes())
        for n in nodes:
            if n != src and n not in dst_set:
                if tree_G.out_degree(n) == 0:
                    tree_G.remove_node(n)
                    changed = True
    return tree_G

def optimize_with_msa(tree_G, G, src, dsts, apsp_cost, apsp_path):
    if tree_G is None:
        return None, float('inf')
        
    tree_nodes = list(tree_G.nodes())
    M = nx.DiGraph()
    for n in tree_nodes:
        M.add_node(n)
        
    for u in tree_nodes:
        for v in tree_nodes:
            if u != v and u in apsp_cost and v in apsp_cost[u]:
                M.add_edge(u, v, weight=apsp_cost[u][v])
                
    try:
        M_rooted = M.copy()
        M_rooted.remove_edges_from(list(M_rooted.in_edges(src)))
        msa = nx.minimum_spanning_arborescence(M_rooted, attr="weight")
    except Exception:
        c = sum(d['cost'] for u, v, d in tree_G.edges(data=True))
        return tree_G, c
        
    opt_G = nx.DiGraph()
    for u, v, data in msa.edges(data=True):
        path = apsp_path[u][v]
        for i in range(len(path) - 1):
            x, y = path[i], path[i+1]
            opt_G.add_edge(x, y, cost=G[x][y]['cost'])
            
    opt_G = prune_leaves(opt_G, src, dsts)
    if opt_G is None:
        c = sum(d['cost'] for u, v, d in tree_G.edges(data=True))
        return tree_G, c
    c = sum(d['cost'] for u, v, d in opt_G.edges(data=True))
    return opt_G, c

def dreyfus_wagner(G, src, dsts, apsp_cost, apsp_path):
    K = len(dsts)
    nodes = list(G.nodes())
    
    D_arr = [[float('inf')] * len(nodes) for _ in range(1 << K)]
    choice_D = [[None] * len(nodes) for _ in range(1 << K)]
    S_arr = [[float('inf')] * len(nodes) for _ in range(1 << K)]
    choice_S = [[None] * len(nodes) for _ in range(1 << K)]
    
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    idx_to_node = {i: n for i, n in enumerate(nodes)}
    
    for i, dst in enumerate(dsts):
        mask = 1 << i
        for v in nodes:
            if v in apsp_cost and dst in apsp_cost[v]:
                D_arr[mask][node_to_idx[v]] = apsp_cost[v][dst]
                choice_D[mask][node_to_idx[v]] = dst

    for k_size in range(2, K + 1):
        for combo in combinations(range(K), k_size):
            mask = 0
            for i in combo:
                mask |= (1 << i)
                
            submask = (mask - 1) & mask
            lsb = mask & -mask
            while submask > 0:
                if (submask & lsb):
                    comp = mask ^ submask
                    for v_idx in range(len(nodes)):
                        cost = D_arr[submask][v_idx] + D_arr[comp][v_idx]
                        if cost < S_arr[mask][v_idx]:
                            S_arr[mask][v_idx] = cost
                            choice_S[mask][v_idx] = submask
                submask = (submask - 1) & mask
                
            for u_idx, u in enumerate(nodes):
                for v_idx, v in enumerate(nodes):
                    if u in apsp_cost and v in apsp_cost[u]:
                        cost = apsp_cost[u][v] + S_arr[mask][v_idx]
                        if cost < D_arr[mask][u_idx]:
                            D_arr[mask][u_idx] = cost
                            choice_D[mask][u_idx] = v
                            
    best_mask = (1 << K) - 1
    best_cost = D_arr[best_mask][node_to_idx[src]]
    
    if best_cost == float('inf'):
        return None, float('inf')
        
    tree_edges = set()
    
    def backtrack_D(mask, u):
        v = choice_D[mask][node_to_idx[u]]
        if v is None: return
        path = apsp_path[u][v]
        for i in range(len(path) - 1):
            tree_edges.add((path[i], path[i+1]))
        if mask & (mask - 1) != 0:
            backtrack_S(mask, v)
            
    def backtrack_S(mask, v):
        submask1 = choice_S[mask][node_to_idx[v]]
        submask2 = mask ^ submask1
        backtrack_D(submask1, v)
        backtrack_D(submask2, v)
        
    backtrack_D(best_mask, src)
    
    tree_G = nx.DiGraph()
    for u, v in tree_edges:
        tree_G.add_edge(u, v, cost=G[u][v]['cost'])
    
    tree_G = prune_leaves(tree_G, src, dsts)
    if tree_G is None:
        return None, float('inf')
        
    final_cost = sum(d['cost'] for u, v, d in tree_G.edges(data=True))
    return tree_G, final_cost

def beam_search(G, src, dsts, apsp_cost, apsp_path, beam_width=30):
    initial_state = (0.0, frozenset([src]), frozenset(dsts), frozenset())
    beam = [initial_state]
    
    for _ in range(len(dsts)):
        next_beam = []
        for cost, conn_nodes, unconn_dsts, tree_edges in beam:
            if not unconn_dsts:
                next_beam.append((cost, conn_nodes, unconn_dsts, tree_edges))
                continue
                
            G_search = nx.DiGraph(G)
            G_search.add_node("DUMMY_SRC")
            for s in conn_nodes:
                G_search.add_edge("DUMMY_SRC", s, cost=0.0)
                
            try:
                lengths, paths = nx.single_source_dijkstra(G_search, "DUMMY_SRC", weight="cost")
            except Exception:
                continue
                
            for dst in unconn_dsts:
                if dst in lengths:
                    path = paths[dst][1:]
                    new_edges = set(tree_edges)
                    new_nodes = set(conn_nodes)
                    
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        new_edges.add((u, v))
                        new_nodes.add(v)
                        
                    new_unconn = set(unconn_dsts)
                    new_unconn.remove(dst)
                    
                    actual_cost = sum(G[u][v]['cost'] for u, v in new_edges)
                    next_beam.append((actual_cost, frozenset(new_nodes), frozenset(new_unconn), frozenset(new_edges)))
                    
        if not next_beam:
            break
            
        unique_next = {}
        for state in next_beam:
            c, n, u, e = state
            if e not in unique_next or c < unique_next[e][0]:
                unique_next[e] = state
                
        next_beam = list(unique_next.values())
        next_beam.sort(key=lambda x: x[0])
        beam = next_beam[:beam_width]
        
    best_tree = None
    best_cost = float('inf')
    
    for cost, conn_nodes, unconn_dsts, tree_edges in beam:
        if not unconn_dsts:
            tree_G = nx.DiGraph()
            for u, v in tree_edges:
                tree_G.add_edge(u, v, cost=G[u][v]['cost'])
            
            tree_G = prune_leaves(tree_G, src, dsts)
            if tree_G is not None:
                c = sum(d['cost'] for u, v, d in tree_G.edges(data=True))
                if c < best_cost:
                    best_cost = c
                    best_tree = tree_G
                
    return best_tree, best_cost

def dijkstra_tm(G, src, dsts, apsp_cost, apsp_path, trials=1000):
    best_cost = float('inf')
    best_tree = None
    
    for trial in range(trials):
        conn_nodes = {src}
        unconn_dsts = set(dsts)
        tree_edges = set()
        
        while unconn_dsts:
            G_search = nx.DiGraph(G)
            G_search.add_node("DUMMY_SRC")
            for s in conn_nodes:
                G_search.add_edge("DUMMY_SRC", s, cost=0.0)
                
            if trial > 0:
                intensity = min(2.0, 1.0 + (trial / 500.0))
                for u, v, d in G_search.edges(data=True):
                    if u != "DUMMY_SRC" and "cost" in d and d["cost"] is not None:
                        d["cost"] = d["cost"] * random.uniform(1.0, intensity)
                        
            try:
                lengths, paths = nx.single_source_dijkstra(G_search, "DUMMY_SRC", weight="cost")
            except Exception:
                break
                
            options = []
            for dst in unconn_dsts:
                if dst in lengths:
                    options.append((lengths[dst], dst, paths[dst]))
                    
            if not options:
                break
                
            options.sort(key=lambda x: x[0])
            
            if trial == 0:
                choice = options[0]
            else:
                top_k = min(len(options), random.randint(1, 3))
                choice = random.choice(options[:top_k])
                
            _, dst, path = choice
            path = path[1:]
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                tree_edges.add((u, v))
                conn_nodes.add(v)
                
            unconn_dsts.remove(dst)
            
        if not unconn_dsts:
            tree_G = nx.DiGraph()
            for u, v in tree_edges:
                tree_G.add_edge(u, v, cost=G[u][v]['cost'])
            
            tree_G = prune_leaves(tree_G, src, dsts)
            if tree_G is not None:
                c = sum(d['cost'] for u, v, d in tree_G.edges(data=True))
                if c < best_cost:
                    best_cost = c
                    best_tree = tree_G
                
    return best_tree, best_cost

def search_algorithm(src, dsts, G, num_partitions):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    
    # Filter out missing costs
    edges_to_remove = []
    for u, v, d in h.edges(data=True):
        if d.get('cost') is None:
            edges_to_remove.append((u, v))
    h.remove_edges_from(edges_to_remove)
    
    apsp = dict(nx.all_pairs_dijkstra(h, weight="cost"))
    apsp_cost = {u: {v: apsp[u][0][v] for v in apsp[u][0]} for u in apsp}
    apsp_path = {u: {v: apsp[u][1][v] for v in apsp[u][1]} for u in apsp}
    
    K = len(dsts)
    best_tree = None
    best_cost = float('inf')
    
    if K <= 12:
        try:
            tree_dw, cost_dw = dreyfus_wagner(h, src, dsts, apsp_cost, apsp_path)
            if tree_dw is not None and cost_dw < best_cost:
                best_cost = cost_dw
                best_tree = tree_dw
        except Exception as e:
            pass
            
    if best_tree is None or best_cost == float('inf'):
        # Beam search
        tree_bs, cost_bs = beam_search(h, src, dsts, apsp_cost, apsp_path, beam_width=30)
        if tree_bs is not None:
            tree_bs_opt, cost_bs_opt = optimize_with_msa(tree_bs, h, src, dsts, apsp_cost, apsp_path)
            if cost_bs_opt < best_cost:
                best_cost = cost_bs_opt
                best_tree = tree_bs_opt
            elif cost_bs < best_cost:
                best_cost = cost_bs
                best_tree = tree_bs
            
        # Dijkstra TM
        tree_tm, cost_tm = dijkstra_tm(h, src, dsts, apsp_cost, apsp_path, trials=1000)
        if tree_tm is not None:
            tree_tm_opt, cost_tm_opt = optimize_with_msa(tree_tm, h, src, dsts, apsp_cost, apsp_path)
            if cost_tm_opt < best_cost:
                best_cost = cost_tm_opt
                best_tree = tree_tm_opt
            elif cost_tm < best_cost:
                best_cost = cost_tm
                best_tree = tree_tm
            
    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    
    if best_tree is not None:
        for dst in dsts:
            try:
                path = nx.dijkstra_path(best_tree, src, dst, weight="cost")
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    for j in range(bc_topology.num_partitions):
                        bc_topology.append_dst_partition_path(dst, j, [u, v, h[u][v]])
            except nx.NetworkXNoPath:
                pass
                
    return bc_topology
