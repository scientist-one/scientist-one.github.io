def compute_model_placement(gpu_num, models):
    import math

    def calc_kvpr(placement):
        max_kvpr = 0.0
        for gid, mods in placement.items():
            if not mods: continue
            L = sum(m.req_rate / m.slo for m in mods)
            S = sum(m.model_size for m in mods)
            if GPU_MEM_SIZE - S <= 0:
                return float('inf')
            max_kvpr = max(max_kvpr, L / (GPU_MEM_SIZE - S))
        return max_kvpr

    best_placement = None
    best_T = float('inf')

    try:
        sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo), reverse=True)
        placement = {gpu_id: [] for gpu_id in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]
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
            if best_idx is not None:
                placement[best_idx].append(model)
                weighted_req_rate[best_idx] += model.req_rate / model.slo
                shared_kv[best_idx] -= model.model_size
            else:
                placement = None
                break
        if placement is not None:
            best_placement = placement
            best_T = calc_kvpr(placement)
    except Exception:
        pass

    def check_T(T):
        def try_pack(indices, fit_type='best'):
            bins_w = [0.0] * gpu_num
            bins_s = [0.0] * gpu_num
            placement = {i: [] for i in range(gpu_num)}
            
            for idx in indices:
                m = models[idx]
                w = m.req_rate / m.slo + T * m.model_size
                
                best_bin = -1
                best_val = None
                
                for j in range(gpu_num):
                    if bins_w[j] + w <= T * GPU_MEM_SIZE + 1e-7 and bins_s[j] + m.model_size < GPU_MEM_SIZE:
                        rem = (T * GPU_MEM_SIZE) - (bins_w[j] + w)
                        if fit_type == 'best':
                            if best_val is None or rem < best_val:
                                best_val = rem
                                best_bin = j
                        elif fit_type == 'worst':
                            if best_val is None or rem > best_val:
                                best_val = rem
                                best_bin = j
                        elif fit_type == 'first':
                            best_bin = j
                            break
                
                if best_bin != -1:
                    j = best_bin
                    bins_w[j] += w
                    bins_s[j] += m.model_size
                    placement[j].append(m)
                else:
                    return None
            return placement

        sorts = [
            sorted(range(len(models)), key=lambda i: models[i].req_rate / models[i].slo + T * models[i].model_size, reverse=True),
            sorted(range(len(models)), key=lambda i: models[i].model_size, reverse=True),
            sorted(range(len(models)), key=lambda i: models[i].req_rate / models[i].slo, reverse=True),
        ]
        
        for indices in sorts:
            for fit in ['best', 'first', 'worst']:
                p = try_pack(indices, fit)
                if p: return p
                
        steps = 0
        max_steps = 2000
        bins_w = [0.0] * gpu_num
        bins_s = [0.0] * gpu_num
        placement = [[] for _ in range(gpu_num)]
        sorted_models = [models[i] for i in sorts[0]]
        
        def backtrack(idx):
            nonlocal steps
            steps += 1
            if steps > max_steps:
                return False
            if idx == len(sorted_models):
                return True
                
            m = sorted_models[idx]
            w = m.req_rate / m.slo + T * m.model_size
            
            empty_tried = False
            for j in range(gpu_num):
                is_empty = (len(placement[j]) == 0)
                if is_empty:
                    if empty_tried:
                        continue
                    empty_tried = True
                
                if bins_w[j] + w <= T * GPU_MEM_SIZE + 1e-7 and bins_s[j] + m.model_size < GPU_MEM_SIZE:
                    bins_w[j] += w
                    bins_s[j] += m.model_size
                    placement[j].append(m)
                    
                    if backtrack(idx + 1):
                        return True
                    
                    bins_w[j] -= w
                    bins_s[j] -= m.model_size
                    placement[j].pop()
            return False
            
        if backtrack(0):
            return {i: placement[i] for i in range(gpu_num)}
            
        return None

    low = 0.0
    high = best_T if best_T != float('inf') else 1e6

    if best_T == float('inf'):
        p = check_T(high)
        while not p and high < 1e12:
            high *= 10
            p = check_T(high)
        if not p:
            raise ValueError("Unable to place models on given GPUs")
        best_placement = p
        best_T = calc_kvpr(p)
        high = best_T

    for _ in range(50):
        mid = (low + high) / 2.0
        p = check_T(mid)
        if p:
            best_placement = p
            high = mid
            best_T = mid
        else:
            low = mid

    if best_placement is None:
        raise ValueError("Unable to place models on given GPUs")
        
    return best_placement
