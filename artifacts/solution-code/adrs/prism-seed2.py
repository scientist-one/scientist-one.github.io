def compute_model_placement(gpu_num, models):
    import random
    
    # Baseline Greedy Placement
    sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo), reverse=True)
    
    M = GPU_MEM_SIZE
    
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    shared_kv = [M for _ in range(gpu_num)]
    weighted_req_rate = [0.0 for _ in range(gpu_num)]
    
    for model in sorted_models:
        best_idx = None
        best_ratio = float('inf')
        for gpu_id in range(gpu_num):
            if model.model_size <= shared_kv[gpu_id] and shared_kv[gpu_id] > 0:
                current_ratio = weighted_req_rate[gpu_id] / shared_kv[gpu_id]
                if current_ratio < best_ratio:
                    best_ratio = current_ratio
                    best_idx = gpu_id
        if best_idx is None:
            raise ValueError("Unable to place model")
        placement[best_idx].append(model)
        weighted_req_rate[best_idx] += model.req_rate / model.slo
        shared_kv[best_idx] -= model.model_size
        
    def get_max_kvpr(plac):
        max_k = 0.0
        for i in range(gpu_num):
            s = sum(m.model_size for m in plac[i])
            w = sum(m.req_rate / m.slo for m in plac[i])
            if s >= M:
                return float('inf')
            max_k = max(max_k, w / (M - s) if plac[i] else 0.0)
        return max_k

    best_placement = placement
    best_kvpr = get_max_kvpr(placement)
    
    # Bisection Search for Target KVPR
    low = 0.0
    high = best_kvpr
    
    for _ in range(60):
        mid = (low + high) / 2.0
        
        strategies = [
            sorted(models, key=lambda m: (m.req_rate / m.slo) + mid * m.model_size, reverse=True),
            sorted(models, key=lambda m: (m.req_rate / m.slo), reverse=True),
            sorted(models, key=lambda m: m.model_size, reverse=True),
            sorted(models, key=lambda m: (m.req_rate / m.slo) / m.model_size, reverse=True)
        ]
        
        for _ in range(15):
            shuffled = list(models)
            random.shuffle(shuffled)
            strategies.append(shuffled)
            
        found_feasible = False
        
        for strat in strategies:
            for use_best_fit in [False, True]:
                temp_plac = {i: [] for i in range(gpu_num)}
                bins_weight = [0.0 for _ in range(gpu_num)]
                bins_size = [0.0 for _ in range(gpu_num)]
                
                feasible = True
                for model in strat:
                    w_i = model.req_rate / model.slo
                    s_i = model.model_size
                    
                    best_idx = -1
                    best_rem = float('inf')
                    
                    for i in range(gpu_num):
                        if (bins_weight[i] + w_i) + mid * (bins_size[i] + s_i) <= mid * M + 1e-9 and bins_size[i] + s_i < M:
                            rem = mid * M - ((bins_weight[i] + w_i) + mid * (bins_size[i] + s_i))
                            if use_best_fit:
                                if rem < best_rem:
                                    best_rem = rem
                                    best_idx = i
                            else:
                                best_idx = i
                                break
                                
                    if best_idx == -1:
                        feasible = False
                        break
                        
                    temp_plac[best_idx].append(model)
                    bins_weight[best_idx] += w_i
                    bins_size[best_idx] += s_i
                    
                if feasible:
                    found_feasible = True
                    best_placement = temp_plac
                    break
            if found_feasible:
                break
                
        if found_feasible:
            high = mid
        else:
            low = mid
            
    # Local Search with optimized variables
    placement_sizes = [sum(m.model_size for m in best_placement[i]) for i in range(gpu_num)]
    placement_weights = [sum(m.req_rate / m.slo for m in best_placement[i]) for i in range(gpu_num)]
    
    for _ in range(1000):
        max_idx = -1
        max_kvpr = -1
        for i in range(gpu_num):
            k = placement_weights[i] / (M - placement_sizes[i]) if best_placement[i] else 0.0
            if k > max_kvpr:
                max_kvpr = k
                max_idx = i
                
        other_max = -1
        for i in range(gpu_num):
            if i != max_idx:
                k = placement_weights[i] / (M - placement_sizes[i]) if best_placement[i] else 0.0
                other_max = max(other_max, k)
                
        best_new_max = max_kvpr
        best_move = None
        
        for model_idx, model in enumerate(best_placement[max_idx]):
            m_s = model.model_size
            m_w = model.req_rate / model.slo
            
            src_s = placement_sizes[max_idx] - m_s
            src_w = placement_weights[max_idx] - m_w
            new_src_k = src_w / (M - src_s) if src_s > 0 else 0.0
            
            for target_idx in range(gpu_num):
                if target_idx == max_idx:
                    continue
                
                # Try Move
                tgt_s = placement_sizes[target_idx] + m_s
                if tgt_s < M:
                    tgt_w = placement_weights[target_idx] + m_w
                    new_tgt_k = tgt_w / (M - tgt_s)
                    
                    new_max = max(new_src_k, new_tgt_k, other_max)
                    if new_max < best_new_max - 1e-9:
                        best_new_max = new_max
                        best_move = ('move', model_idx, target_idx, m_s, m_w)
                
                # Try Swap
                for tgt_model_idx, tgt_model in enumerate(best_placement[target_idx]):
                    tm_s = tgt_model.model_size
                    tm_w = tgt_model.req_rate / tgt_model.slo
                    
                    swap_src_s = src_s + tm_s
                    swap_tgt_s = placement_sizes[target_idx] - tm_s + m_s
                    
                    if swap_src_s >= M or swap_tgt_s >= M:
                        continue
                        
                    swap_src_w = src_w + tm_w
                    swap_tgt_w = placement_weights[target_idx] - tm_w + m_w
                    
                    new_swap_src_k = swap_src_w / (M - swap_src_s)
                    new_swap_tgt_k = swap_tgt_w / (M - swap_tgt_s)
                    
                    new_max = max(new_swap_src_k, new_swap_tgt_k, other_max)
                    if new_max < best_new_max - 1e-9:
                        best_new_max = new_max
                        best_move = ('swap', model_idx, target_idx, tgt_model_idx, m_s, m_w, tm_s, tm_w)
                        
        if best_move is None:
            break
            
        if best_move[0] == 'move':
            _, m_idx, t_idx, m_s, m_w = best_move
            best_placement[t_idx].append(best_placement[max_idx].pop(m_idx))
            placement_sizes[max_idx] -= m_s
            placement_weights[max_idx] -= m_w
            placement_sizes[t_idx] += m_s
            placement_weights[t_idx] += m_w
        else:
            _, m_idx, t_idx, tm_idx, m_s, m_w, tm_s, tm_w = best_move
            m1 = best_placement[max_idx].pop(m_idx)
            m2 = best_placement[t_idx].pop(tm_idx)
            best_placement[max_idx].append(m2)
            best_placement[t_idx].append(m1)
            placement_sizes[max_idx] = placement_sizes[max_idx] - m_s + tm_s
            placement_weights[max_idx] = placement_weights[max_idx] - m_w + tm_w
            placement_sizes[t_idx] = placement_sizes[t_idx] - tm_s + m_s
            placement_weights[t_idx] = placement_weights[t_idx] - tm_w + m_w
            
    return best_placement
