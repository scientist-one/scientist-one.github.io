def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    """
    import copy
    
    # Assuming GPU_MEM_SIZE is in globals
    M = GPU_MEM_SIZE
    
    def get_R(m):
        return m.req_rate / m.slo
        
    def get_S(m):
        return m.model_size
        
    total_R = sum(get_R(m) for m in models)
    total_S = sum(get_S(m) for m in models)
    
    if total_S > gpu_num * M:
        raise ValueError("Total model size exceeds total GPU memory")
        
    low = 0
    if gpu_num * M - total_S > 0:
        low = total_R / (gpu_num * M - total_S)
    high = 1e9
    
    def get_heuristics(lambda_val):
        return [
            lambda m: get_R(m) + lambda_val * get_S(m),
            lambda m: get_S(m),
            lambda m: get_R(m),
            lambda m: (get_R(m) + lambda_val * get_S(m)) / get_S(m) if get_S(m) > 0 else 0,
            lambda m: get_R(m) / get_S(m) if get_S(m) > 0 else 0,
        ]

    def try_pack(lambda_val):
        cap = lambda_val * M
        
        # 1. EXACT SEARCH with step limit (for small enough instances or quick pruning)
        models_list = sorted(models, key=lambda m: get_S(m), reverse=True)
        bins_R = [0] * gpu_num
        bins_S = [0] * gpu_num
        assignment = [-1] * len(models)
        steps = 0
        hit_limit = False
        
        def search(idx):
            nonlocal steps, hit_limit
            steps += 1
            if steps > 5000:
                hit_limit = True
                return False
            if idx == len(models):
                return True
            
            m = models_list[idx]
            R_m = get_R(m)
            S_m = get_S(m)
            
            for i in range(gpu_num):
                # Symmetry breaking: only try the first empty bin
                if bins_S[i] == 0:
                    first_empty = True
                    for j in range(i):
                        if bins_S[j] == 0:
                            first_empty = False
                            break
                    if not first_empty:
                        continue
                        
                if bins_S[i] + S_m <= M and (bins_R[i] + R_m) + lambda_val * (bins_S[i] + S_m) <= cap + 1e-9:
                    bins_S[i] += S_m
                    bins_R[i] += R_m
                    assignment[idx] = i
                    if search(idx + 1):
                        return True
                    bins_S[i] -= S_m
                    bins_R[i] -= R_m
                    if hit_limit:
                        return False
            return False

        if search(0):
            res = {i: [] for i in range(gpu_num)}
            for idx, i in enumerate(assignment):
                res[i].append(models_list[idx])
            return res
            
        if not hit_limit:
            return None # proven no solution
            
        # 2. Heuristics fallback
        for key_func in get_heuristics(lambda_val):
            sorted_models = sorted(models, key=key_func, reverse=True)
            
            # First Fit
            bins = [{"R": 0, "S": 0, "models": []} for _ in range(gpu_num)]
            success = True
            for m in sorted_models:
                R_m = get_R(m)
                S_m = get_S(m)
                placed = False
                for b in bins:
                    if b["S"] + S_m <= M and (b["R"] + R_m) + lambda_val * (b["S"] + S_m) <= cap + 1e-9:
                        b["S"] += S_m
                        b["R"] += R_m
                        b["models"].append(m)
                        placed = True
                        break
                if not placed:
                    success = False
                    break
            if success:
                return {i: bins[i]["models"] for i in range(gpu_num)}
                
            # Best Fit (min remaining cap)
            bins = [{"R": 0, "S": 0, "models": []} for _ in range(gpu_num)]
            success = True
            for m in sorted_models:
                R_m = get_R(m)
                S_m = get_S(m)
                best_bin_idx = -1
                best_rem = float('inf')
                for i, b in enumerate(bins):
                    if b["S"] + S_m <= M and (b["R"] + R_m) + lambda_val * (b["S"] + S_m) <= cap + 1e-9:
                        rem = cap - ((b["R"] + R_m) + lambda_val * (b["S"] + S_m))
                        if rem < best_rem:
                            best_rem = rem
                            best_bin_idx = i
                if best_bin_idx != -1:
                    bins[best_bin_idx]["S"] += S_m
                    bins[best_bin_idx]["R"] += R_m
                    bins[best_bin_idx]["models"].append(m)
                else:
                    success = False
                    break
            if success:
                return {i: bins[i]["models"] for i in range(gpu_num)}
                
            # Worst Fit (max remaining cap)
            bins = [{"R": 0, "S": 0, "models": []} for _ in range(gpu_num)]
            success = True
            for m in sorted_models:
                R_m = get_R(m)
                S_m = get_S(m)
                best_bin_idx = -1
                best_rem = -float('inf')
                for i, b in enumerate(bins):
                    if b["S"] + S_m <= M and (b["R"] + R_m) + lambda_val * (b["S"] + S_m) <= cap + 1e-9:
                        rem = cap - ((b["R"] + R_m) + lambda_val * (b["S"] + S_m))
                        if rem > best_rem:
                            best_rem = rem
                            best_bin_idx = i
                if best_bin_idx != -1:
                    bins[best_bin_idx]["S"] += S_m
                    bins[best_bin_idx]["R"] += R_m
                    bins[best_bin_idx]["models"].append(m)
                else:
                    success = False
                    break
            if success:
                return {i: bins[i]["models"] for i in range(gpu_num)}
                
        return None

    # Baseline placement for valid upper bound
    best_placement = None
    best_lambda = float('inf')
    
    sorted_models_baseline = sorted(models, key=lambda m: get_R(m), reverse=True)
    baseline_placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    shared_kv = [M for _ in range(gpu_num)]
    weighted_req_rate = [0.0 for _ in range(gpu_num)]
    
    baseline_success = True
    for model in sorted_models_baseline:
        best_idx = None
        best_ratio = float('inf')
        for gpu_id in range(gpu_num):
            if get_S(model) <= shared_kv[gpu_id] and shared_kv[gpu_id] > 0:
                current_ratio = weighted_req_rate[gpu_id] / shared_kv[gpu_id]
                if current_ratio < best_ratio:
                    best_ratio = current_ratio
                    best_idx = gpu_id
        if best_idx is None:
            baseline_success = False
            break
        baseline_placement[best_idx].append(model)
        weighted_req_rate[best_idx] += get_R(model)
        shared_kv[best_idx] -= get_S(model)
        
    if baseline_success:
        max_kvpr = 0
        for i in range(gpu_num):
            if M - shared_kv[i] > 0:
                if shared_kv[i] > 0:
                    kvpr = weighted_req_rate[i] / shared_kv[i]
                else:
                    kvpr = float('inf') if weighted_req_rate[i] > 0 else 0
                if kvpr > max_kvpr:
                    max_kvpr = kvpr
        best_placement = baseline_placement
        best_lambda = max_kvpr
        high = min(high, max_kvpr)

    # Binary search
    for _ in range(60):
        mid = (low + high) / 2
        res = try_pack(mid)
        if res is not None:
            best_placement = res
            best_lambda = mid
            high = mid
        else:
            low = mid
            
    if best_placement is None:
        # Fallback to extremely large lambda
        res = try_pack(1e9)
        if res is not None:
            return res
        raise ValueError("Could not find a valid placement")
        
    return best_placement
