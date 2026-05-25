import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List
import time
import math
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
        self.paths[dst][str(partition)] = paths
        
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
        if edge[-1]["cost"] is None:
            no_cost_pairs.append((edge[0], edge[1]))
    return G

class MCTSNode:
    def __init__(self, parent_edges, unreached, parent_node=None, action=None):
        self.parent_edges = parent_edges
        self.unreached = unreached
        self.parent_node = parent_node
        self.action = action
        self.children = {}
        self.visits = 0
        self.sum_cost = 0.0
        self.untried_actions = list(unreached)
        
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
        
    def is_terminal(self):
        return len(self.unreached) == 0

def get_tree_cost(parent_dict, h):
    return sum(h[u][v]['cost'] for v, u in parent_dict.items())

def search_algorithm(src, dsts, G, num_partitions):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    
    global_best_cost = float('inf')
    global_best_tree = None

    def update_best(cost, parent_dict):
        nonlocal global_best_cost, global_best_tree
        if cost < global_best_cost:
            global_best_cost = cost
            global_best_tree = dict(parent_dict)

    root = MCTSNode(frozenset(), frozenset(dsts))
    
    start_time = time.time()
    time_limit = 2.0 
    
    def rollout(parent_edges, unreached_set):
        parent_dict = dict(parent_edges)
        unreached = set(unreached_set)
        
        while unreached:
            V_T = set(parent_dict.keys())
            V_T.add(src)
            
            lengths, paths = nx.multi_source_dijkstra(h, V_T, weight="cost")
            
            candidates = []
            for d in unreached:
                if d in lengths:
                    candidates.append((lengths[d], d, paths[d]))
                    
            if not candidates:
                break
                
            candidates.sort(key=lambda x: x[0])
            top_k = min(3, len(candidates))
            chosen = random.choice(candidates[:top_k])
            chosen_path = chosen[2]
            
            for i in range(1, len(chosen_path)):
                u = chosen_path[i-1]
                v = chosen_path[i]
                if v not in parent_dict and v != src:
                    parent_dict[v] = u
                if v in unreached:
                    unreached.remove(v)
                    
        cost = get_tree_cost(parent_dict, h)
        if not unreached:
            update_best(cost, parent_dict)
        return cost

    iters = 0
    while time.time() - start_time < time_limit:
        iters += 1
        node = root
        
        # Selection
        path = [node]
        while node.is_fully_expanded() and not node.is_terminal():
            best_score = -float('inf')
            best_child = None
            for child in node.children.values():
                if child.visits == 0:
                    continue
                avg_cost = child.sum_cost / child.visits
                explore = math.sqrt(2 * math.log(node.visits) / child.visits)
                score = -avg_cost + 100 * explore
                if score > best_score:
                    best_score = score
                    best_child = child
            if best_child is not None:
                node = best_child
                path.append(node)
            else:
                break

        # Expansion
        if not node.is_terminal() and not node.is_fully_expanded():
            action = node.untried_actions.pop()
            
            parent_dict = dict(node.parent_edges)
            V_T = set(parent_dict.keys())
            V_T.add(src)
            
            try:
                length, chosen_path = nx.multi_source_dijkstra(h, V_T, action, weight="cost")
                
                unreached = set(node.unreached)
                for i in range(1, len(chosen_path)):
                    u = chosen_path[i-1]
                    v = chosen_path[i]
                    if v not in parent_dict and v != src:
                        parent_dict[v] = u
                    if v in unreached:
                        unreached.remove(v)
                
                new_state_edges = frozenset(parent_dict.items())
                new_unreached = frozenset(unreached)
                
                child_node = MCTSNode(new_state_edges, new_unreached, parent_node=node, action=action)
                node.children[action] = child_node
                node = child_node
                path.append(node)
            except nx.NetworkXNoPath:
                pass

        # Rollout
        cost = rollout(node.parent_edges, node.unreached)
        
        # Backpropagation
        for n in path:
            n.visits += 1
            n.sum_cost += cost
            
    if global_best_tree is None:
        global_best_tree = {}
        for dst in dsts:
            chosen_path = nx.dijkstra_path(h, src, dst, weight="cost")
            for i in range(1, len(chosen_path)):
                if chosen_path[i] not in global_best_tree:
                    global_best_tree[chosen_path[i]] = chosen_path[i-1]
                    
    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    
    def get_path_to_node(target):
        p = []
        curr = target
        while curr != src:
            prev = global_best_tree[curr]
            p.append((prev, curr))
            curr = prev
        p.reverse()
        return p

    for dst in dsts:
        edges = get_path_to_node(dst)
        for u, v in edges:
            for j in range(bc_topology.num_partitions):
                bc_topology.append_dst_partition_path(dst, j, [u, v, h[u][v]])

    return bc_topology
