import numpy as np
import random

def get_best_schedule(workload, num_seqs):
    N = workload.num_txns
    
    _cost_cache = {}
    def get_cost(seq):
        t_seq = tuple(seq)
        if t_seq not in _cost_cache:
            _cost_cache[t_seq] = workload.get_opt_seq_cost(list(seq))
        return _cost_cache[t_seq]

    W = np.zeros((N, N))
    single_costs = [get_cost([i]) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            c_ij = get_cost([i, j])
            c_ji = get_cost([j, i])
            
            if c_ij < c_ji:
                W[j, i] = c_ji - c_ij
            elif c_ji < c_ij:
                W[i, j] = c_ij - c_ji
            else:
                delay = c_ij - max(single_costs[i], single_costs[j])
                if delay > 0:
                    W[i, j] = delay / 2.0
                    W[j, i] = delay / 2.0

    row_sums = W.sum(axis=1)
    for i in range(N):
        if row_sums[i] > 0:
            W[i, :] = W[i, :] / row_sums[i]
        else:
            W[i, :] = 1.0 / N

    pr = np.ones(N) / N
    d = 0.85
    for _ in range(50):
        new_pr = (1 - d) / N + d * W.T.dot(pr)
        if np.abs(new_pr - pr).sum() < 1e-6:
            pr = new_pr
            break
        pr = new_pr

    pr_sorted = np.argsort(pr)[::-1]
    
    starts = list(pr_sorted[:25]) + random.sample(range(N), min(10, N))
    starts = list(set(starts))
    
    best_overall_cost = float('inf')
    best_overall_seq = []
    
    for start_t in starts:
        seq = [start_t]
        remaining = set(range(N))
        remaining.remove(start_t)
        
        while remaining:
            rem_list = list(remaining)
            rem_pr = [(t, pr[t]) for t in rem_list]
            rem_pr.sort(key=lambda x: x[1], reverse=True)
            candidates = [t for t, _ in rem_pr[:30]]
            if len(rem_list) > 30:
                random_cands = random.sample([t for t in rem_list if t not in candidates], min(5, len(rem_list) - len(candidates)))
                candidates.extend(random_cands)
            
            cand_costs = []
            for t in candidates:
                cand_costs.append((get_cost(seq + [t]), t))
            cand_costs.sort()
            
            top_cands = cand_costs[:10]
            best_t = top_cands[0][1]
            
            if len(remaining) > 1:
                min_2_cost = float('inf')
                for _, t in top_cands:
                    c2_cands = [x for x in candidates if x != t][:10]
                    if not c2_cands:
                        c2_cands = [x for x in rem_list if x != t][:10]
                    for t2 in c2_cands:
                        c2 = get_cost(seq + [t, t2])
                        if c2 < min_2_cost:
                            min_2_cost = c2
                            best_t = t
            
            seq.append(best_t)
            remaining.remove(best_t)
            
        c = get_cost(seq)
        if c < best_overall_cost:
            best_overall_cost = c
            best_overall_seq = seq.copy()

    improved = True
    best_seq = best_overall_seq.copy()
    best_cost = best_overall_cost
    
    while improved:
        improved = False
        
        for i in range(N):
            for j in range(max(0, i - 30), min(N, i + 31)):
                if i == j: continue
                new_seq = best_seq.copy()
                t = new_seq.pop(i)
                new_seq.insert(j, t)
                c = get_cost(new_seq)
                if c < best_cost:
                    best_cost = c
                    best_seq = new_seq
                    improved = True
                    break
            if improved: break
        if improved: continue
        
        for i in range(N):
            for j in range(i + 1, min(N, i + 21)):
                new_seq = best_seq.copy()
                new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                c = get_cost(new_seq)
                if c < best_cost:
                    best_cost = c
                    best_seq = new_seq
                    improved = True
                    break
            if improved: break
        if improved: continue

        for _ in range(150):
            i, j = random.sample(range(N), 2)
            new_seq = best_seq.copy()
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
            c = get_cost(new_seq)
            if c < best_cost:
                best_cost = c
                best_seq = new_seq
                improved = True
                break
            
    return best_cost, best_seq
