import random

def get_best_schedule(workload, num_seqs):
    beam_width = 40
    sample_size = 50
    
    cost_cache = {}
    def get_cost(seq):
        seq_tup = tuple(seq)
        if seq_tup not in cost_cache:
            cost_cache[seq_tup] = workload.get_opt_seq_cost(seq)
        return cost_cache[seq_tup]

    beam = [(0, [])]
    
    for step in range(workload.num_txns):
        new_candidates = []
        for cost, seq in beam:
            remaining = [t for t in range(workload.num_txns) if t not in seq]
            
            if len(remaining) > sample_size:
                candidates = random.sample(remaining, sample_size)
            else:
                candidates = remaining
                
            for t in candidates:
                new_seq = seq + [t]
                new_cost = get_cost(new_seq)
                new_candidates.append((new_cost, new_seq))
                
        new_candidates.sort(key=lambda x: x[0])
        beam = new_candidates[:beam_width]

    best_overall_cost = float('inf')
    best_overall_seq = []
    
    for initial_cost, initial_seq in beam[:4]:
        best_cost = initial_cost
        best_txn_seq = initial_seq[:]
        
        improved = True
        while improved:
            improved = False
            for _ in range(1500):
                i = random.randint(0, workload.num_txns - 1)
                j = random.randint(0, workload.num_txns - 1)
                if i == j: continue
                
                new_seq = best_txn_seq[:]
                r = random.random()
                if r < 0.33:
                    new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                elif r < 0.66:
                    val = new_seq.pop(i)
                    new_seq.insert(j, val)
                else:
                    if i > j: i, j = j, i
                    new_seq[i:j+1] = reversed(new_seq[i:j+1])
                    
                new_cost = get_cost(new_seq)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_txn_seq = new_seq
                    improved = True
                    break
                    
        if best_cost < best_overall_cost:
            best_overall_cost = best_cost
            best_overall_seq = best_txn_seq

    return best_overall_cost, best_overall_seq
