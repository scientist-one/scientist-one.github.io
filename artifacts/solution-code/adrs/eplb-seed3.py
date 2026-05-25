import torch

def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    num_layers, num_groups = weight.shape
    device = weight.device
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(weight.size(-1),
                                  dtype=torch.int64,
                                  device=device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    sorted_indices = weight.float().sort(-1, descending=True).indices
    
    p_idx_base = torch.arange(num_packs, device=device)
    p_idx_rev = torch.arange(num_packs - 1, -1, -1, device=device)
    
    p_idx_layers = []
    for k in range(groups_per_pack):
        if k % 2 == 0:
            p_idx_layers.append(p_idx_base)
        else:
            p_idx_layers.append(p_idx_rev)
            
    p_idx_seq = torch.cat(p_idx_layers)
    
    pack_index = torch.zeros((num_layers, num_groups), dtype=torch.int64, device=device)
    rank_seq = torch.arange(groups_per_pack, device=device).repeat_interleave(num_packs)
    rank_in_pack = torch.zeros((num_layers, num_groups), dtype=torch.int64, device=device)
    
    L_idx = torch.arange(num_layers, device=device).unsqueeze(1)
    pack_index[L_idx, sorted_indices] = p_idx_seq
    rank_in_pack[L_idx, sorted_indices] = rank_seq
    
    return pack_index, rank_in_pack

def replicate_experts(weight: torch.Tensor, num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    
    if num_redundant == 0:
        return phy2log, rank, logcnt
        
    divisors = torch.arange(1, num_redundant + 1, device=device).view(1, 1, num_redundant)
    quotients = weight.float().unsqueeze(-1) / divisors
    
    epsilon = torch.arange(num_redundant, device=device).float().view(1, 1, num_redundant) * 1e-7
    quotients = quotients - epsilon
    
    quotients_flat = quotients.view(n, -1)
    topk_indices = quotients_flat.topk(num_redundant, dim=-1).indices
    
    selected_log = topk_indices // num_redundant
    selected_ranks = (topk_indices % num_redundant) + 1
    
    phy2log[:, num_log:] = selected_log
    rank[:, num_log:] = selected_ranks
    
    ones = torch.ones_like(selected_log)
    logcnt.scatter_add_(1, selected_log, ones)
    
    return phy2log, rank, logcnt

def rebalance_experts_hierarchical(weight: torch.Tensor, num_physical_experts: int, num_groups: int, num_nodes: int, num_gpus: int):
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(1, perm, torch.arange(perm.size(1), dtype=torch.int64, device=perm.device).expand(perm.shape))
        return inv

    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) * group_size).unsqueeze(-1) + torch.arange(group_size, dtype=torch.int64, device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    tokens_per_mlog = weight.gather(-1, mlog2log).view(-1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(tokens_per_mlog, num_physical_experts // num_nodes)

    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(-1, pphy2phy)
    pphy2mlog = (pphy2mlog.view(num_layers, num_nodes, -1) + torch.arange(0, num_logical_experts, num_logical_experts // num_nodes, device=group_pack_index.device).view(1, -1, 1)).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
    return pphy2log, pphyrank, logcnt

def rebalance_experts(weight: torch.Tensor, num_replicas: int, num_groups: int, num_nodes: int, num_gpus: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_layers, num_logical_experts = weight.shape
    weight = weight.float()
    if num_groups % num_nodes == 0:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(weight, num_replicas, 1, 1, num_gpus)
    num_redundant_experts = num_replicas - num_logical_experts
    maxlogcnt = num_redundant_experts + 1
    log2phy = torch.full((num_layers, num_logical_experts, maxlogcnt), -1, dtype=torch.int64, device=logcnt.device)
    log2phy.view(num_layers, -1).scatter_(-1, phy2log * maxlogcnt + phyrank, torch.arange(num_replicas, dtype=torch.int64, device=log2phy.device).expand(num_layers, -1))
    return phy2log, log2phy, logcnt
