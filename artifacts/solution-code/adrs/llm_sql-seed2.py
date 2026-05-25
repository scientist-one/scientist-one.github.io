import pandas as pd
from typing import Tuple, List, Dict
import numpy as np

try:
    from solver import Algorithm
except ImportError:
    class Algorithm:
        pass

class Evolved(Algorithm):
    def __init__(self, df: pd.DataFrame = None):
        self.df = df

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
        
        cols = list(df.columns)
        num_rows = len(df)
        
        col_values = {}
        col_lensq = {}
        
        for c in cols:
            series = df[c]
            raw_list = series.tolist()
            
            processed_vals = []
            lensq_vals = []
            
            for val in raw_list:
                try:
                    hash(val)
                    pval = val
                except TypeError:
                    pval = str(val)
                
                if isinstance(val, str):
                    l = len(val)
                else:
                    l = len(str(val))
                    
                processed_vals.append(pval)
                lensq_vals.append(l * l)
                
            col_values[c] = processed_vals
            col_lensq[c] = lensq_vals

        def build_permutations(R: List[int], C: List[str]) -> List[Tuple[int, List[str]]]:
            if not C or len(R) <= 1:
                return [(r, list(C)) for r in R]
                
            best_c = None
            best_score = -1.0
            
            for c in C:
                c_vals = col_values[c]
                c_lensq = col_lensq[c]
                
                seen = set()
                score = 0.0
                
                for r in R:
                    val = c_vals[r]
                    if val in seen:
                        score += c_lensq[r]
                    else:
                        seen.add(val)
                        
                if score > best_score:
                    best_score = score
                    best_c = c
                    
            if best_score <= 0.0:
                return [(r, list(C)) for r in R]
                
            sub_C = [c for c in C if c != best_c]
            
            # Reconstruct groups for best_c
            val_to_rows = {}
            c_vals = col_values[best_c]
            for r in R:
                val = c_vals[r]
                if val in val_to_rows:
                    val_to_rows[val].append(r)
                else:
                    val_to_rows[val] = [r]
                    
            best_freq_groups = {v: rows for v, rows in val_to_rows.items() if len(rows) > 1}
            
            results = []
            
            sorted_vals = sorted(best_freq_groups.keys(), key=lambda v: str(v))
            
            R_freq = set()
            for val in sorted_vals:
                rows = best_freq_groups[val]
                sub_results = build_permutations(rows, sub_C)
                for r, order in sub_results:
                    results.append((r, [best_c] + order))
                R_freq.update(rows)
                
            R_rest = [r for r in R if r not in R_freq]
            if R_rest:
                sub_results = build_permutations(R_rest, sub_C)
                for r, order in sub_results:
                    results.append((r, order + [best_c]))
                    
            return results

        all_row_indices = list(range(num_rows))
        results = build_permutations(all_row_indices, cols)
        
        final_row_indices = [r for r, order in results]
        final_col_orders = [order for r, order in results]
        
        df_vals = df.to_numpy()
        col_locs = {c: j for j, c in enumerate(cols)}
        
        new_data = []
        for r, order in zip(final_row_indices, final_col_orders):
            new_data.append([df_vals[r, col_locs[c]] for c in order])
            
        final_df = pd.DataFrame(new_data, columns=cols)
        
        return final_df, final_col_orders
