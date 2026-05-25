def get_best_schedule(workload, num_seqs):
    import random
    
    memo = {}
    def get_cost(seq):
        t = tuple(seq)
        if t not in memo:
            memo[t] = workload.get_opt_seq_cost(list(seq))
        return memo[t]
        
    beam_width = 50
    beam = [([i], get_cost([i])) for i in range(workload.num_txns)]
    beam.sort(key=lambda x: x[1])
    beam = beam[:beam_width]
    
    for step in range(1, workload.num_txns):
        next_beam = []
        visited_states = {}
        
        for seq, _ in beam:
            remaining_txns = set(range(workload.num_txns)) - set(seq)
            for t in remaining_txns:
                new_seq = seq + [t]
                cost = get_cost(new_seq)
                state = frozenset(new_seq)
                
                if state not in visited_states or cost < visited_states[state][1]:
                    visited_states[state] = (new_seq, cost)
                    
        next_beam = list(visited_states.values())
        next_beam.sort(key=lambda x: x[1])
        beam = next_beam[:beam_width]
        
    best_seq, best_cost = beam[0]
    return best_cost, best_seq
