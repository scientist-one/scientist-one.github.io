import pandas as pd
from typing import Tuple, List, Dict
import numpy as np
import heapq
import sys
import os

try:
    from solver import Algorithm
except ImportError:
    sys.path.append(os.path.abspath("benchmarks/ADRS/llm_sql"))
    try:
        from solver import Algorithm
    except ImportError:
        class Algorithm:
            pass

class Evolved(Algorithm):
    def reorder(
        self,
        df: pd.DataFrame,
        early_stop: int = 0,
        row_stop: int = None,
        col_stop: int = None,
        col_merge: List[List[str]] = [],
        one_way_dep: List[Tuple[str, str]] = [],
        distinct_value_threshold: float = 0.8,
        parallel: bool = True,
    ) -> Tuple[pd.DataFrame, List[List[str]]]:
        num_rows, num_cols = df.shape
        if num_rows == 0 or num_cols == 0:
            return df, []
            
        cols = df.columns.tolist()
        
        pair_to_id = {}
        id_to_pair = {}
        id_to_score_multiplier = []
        
        records_id = np.empty((num_rows, num_cols), dtype=np.int32)
        records_val = np.empty((num_rows, num_cols), dtype=object)
        
        df_vals = df.to_numpy()
        
        next_id = 0
        for c_idx in range(num_cols):
            for r_idx in range(num_rows):
                val = df_vals[r_idx, c_idx]
                records_val[r_idx, c_idx] = val
                
                str_val = str(val)
                pair = (c_idx, str_val)
                if pair not in pair_to_id:
                    pair_to_id[pair] = next_id
                    id_to_pair[next_id] = pair
                    length_sq = len(str_val) ** 2
                    id_to_score_multiplier.append(length_sq)
                    next_id += 1
                    
                records_id[r_idx, c_idx] = pair_to_id[pair]
                
        id_to_score_multiplier = np.array(id_to_score_multiplier, dtype=np.float64)
        
        final_orders = []
        final_rows = []
        
        def solve_sub(R_sub, C_sub, path_prefix, current_counts=None, current_heap=None):
            if not R_sub:
                return
            if not C_sub:
                for r in R_sub:
                    final_orders.append(path_prefix[:])
                    final_rows.append(r)
                return
                
            if len(R_sub) == 1:
                for r in R_sub:
                    final_orders.append(path_prefix + C_sub)
                    final_rows.append(r)
                return
            
            if current_counts is None:
                sub_matrix = records_id[R_sub][:, C_sub]
                current_counts = np.bincount(sub_matrix.ravel(), minlength=next_id)
                
                current_heap = []
                valid_mask = current_counts > 1
                valid_ids = np.nonzero(valid_mask)[0]
                for pid in valid_ids:
                    score = (current_counts[pid] - 1) * id_to_score_multiplier[pid]
                    if score > 0:
                        current_heap.append((-score, pid))
                heapq.heapify(current_heap)
            
            while True:
                top_k = []
                K = 5
                
                while current_heap and len(top_k) < K:
                    neg_score, pid = heapq.heappop(current_heap)
                    c, _ = id_to_pair[pid]
                    if c not in C_sub:
                        continue
                    
                    true_count = current_counts[pid]
                    if true_count <= 1:
                        continue
                    
                    true_score = (true_count - 1) * id_to_score_multiplier[pid]
                    if true_score <= 0:
                        continue
                    
                    if true_score == -neg_score:
                        top_k.append((true_score, pid))
                    else:
                        heapq.heappush(current_heap, (-true_score, pid))
                        
                if not top_k:
                    for r in R_sub:
                        final_orders.append(path_prefix + C_sub)
                        final_rows.append(r)
                    return
                
                if len(R_sub) <= 500 or len(top_k) == 1:
                    best_score_1, best_pid = top_k[0]
                    best_c, _ = id_to_pair[best_pid]
                    for i in range(1, len(top_k)):
                        s, p = top_k[i]
                        heapq.heappush(current_heap, (-s, p))
                else:
                    best_total_score = -1.0
                    best_idx = 0
                    
                    for i, (score_1, cand_pid) in enumerate(top_k):
                        cand_c, _ = id_to_pair[cand_pid]
                        R_sub_arr = np.array(R_sub, dtype=np.int32)
                        col_vals = records_id[R_sub_arr, cand_c]
                        mask = (col_vals == cand_pid)
                        R1_arr = R_sub_arr[mask]
                        
                        best_score_2 = 0.0
                        if len(R1_arr) > 1:
                            c_sub_next = [c2 for c2 in C_sub if c2 != cand_c]
                            if c_sub_next:
                                sub_matrix = records_id[R1_arr][:, c_sub_next]
                                counts_2 = np.bincount(sub_matrix.ravel(), minlength=next_id)
                                valid_mask_2 = counts_2 > 1
                                if np.any(valid_mask_2):
                                    scores_2 = (counts_2[valid_mask_2] - 1) * id_to_score_multiplier[valid_mask_2]
                                    if len(scores_2) > 0:
                                        best_score_2 = float(scores_2.max())
                                        
                        total_score = score_1 + best_score_2
                        if total_score > best_total_score:
                            best_total_score = total_score
                            best_idx = i
                            
                    best_score_1, best_pid = top_k[best_idx]
                    best_c, _ = id_to_pair[best_pid]
                    for i in range(len(top_k)):
                        if i != best_idx:
                            s, p = top_k[i]
                            heapq.heappush(current_heap, (-s, p))
                            
                R1 = []
                R2 = []
                for r in R_sub:
                    if records_id[r, best_c] == best_pid:
                        R1.append(r)
                    else:
                        R2.append(r)
                
                if not R2:
                    C_sub_next = [c for c in C_sub if c != best_c]
                    solve_sub(R1, C_sub_next, path_prefix + [best_c], None, None)
                    return
                
                R1_arr = np.array(R1, dtype=np.int32)
                sub_matrix = records_id[R1_arr][:, C_sub]
                counts_to_remove = np.bincount(sub_matrix.ravel(), minlength=next_id)
                current_counts -= counts_to_remove
                
                C_sub_next = [c for c in C_sub if c != best_c]
                solve_sub(R1, C_sub_next, path_prefix + [best_c], None, None)
                
                R_sub = R2

        sys.setrecursionlimit(max(2000, sys.getrecursionlimit()))
        R_all = list(range(num_rows))
        C_all = list(range(num_cols))
        solve_sub(R_all, C_all, [])
        
        reordered_df_rows = []
        final_column_names_orders = []
        for r_idx, order in zip(final_rows, final_orders):
            col_names = [cols[i] for i in order]
            final_column_names_orders.append(col_names)
            row_vals = [records_val[r_idx, i] for i in order]
            reordered_df_rows.append(row_vals)
            
        reordered_df = pd.DataFrame(reordered_df_rows, columns=cols)
        return reordered_df, final_column_names_orders
