import torch

def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(weight.size(-1),
                                  dtype=torch.int64,
                                  device=weight.device).unsqueeze(0).expand(num_layers, num_groups)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    indices = weight.float().sort(-1, descending=True).indices
    
    pack_index = torch.full_like(weight, fill_value=-1, dtype=torch.int64)
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)
    
    pack_weights = torch.zeros(num_layers, num_packs, dtype=torch.float32, device=weight.device)
    pack_items = torch.zeros(num_layers, num_packs, dtype=torch.int64, device=weight.device)
    
    batch_idx = torch.arange(num_layers, device=weight.device)
    
    for j in range(num_groups):
        group = indices[:, j]
        
        is_full = pack_items >= groups_per_pack
        penalty = torch.where(is_full, float('inf'), 0.0)
        
        eff_weights = pack_weights + penalty
        
        pack = eff_weights.argmin(dim=-1)
        
        pack_index[batch_idx, group] = pack
        rank_in_pack[batch_idx, group] = pack_items[batch_idx, pack]
        
        w = weight[batch_idx, group]
        pack_weights[batch_idx, pack] += w
        pack_items[batch_idx, pack] += 1

    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    device = weight.device
    
    if num_redundant <= 0:
        phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).unsqueeze(0).expand(n, num_phy)
        rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
        logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
        return phy2log, rank, logcnt
        
    c = torch.arange(1, num_redundant + 1, dtype=weight.dtype, device=device)
    
    M = weight.unsqueeze(-1) / c.view(1, 1, -1)
    M_flat = M.view(n, -1)
    
    topk_indices = torch.topk(M_flat, num_redundant, dim=-1).indices
    
    expert_indices = topk_indices // num_redundant
    redundant_ranks = (topk_indices % num_redundant) + 1
    
    base_experts = torch.arange(num_log, dtype=torch.int64, device=device).unsqueeze(0).expand(n, num_log)
    base_ranks = torch.zeros(n, num_log, dtype=torch.int64, device=device)
    
    phy2log = torch.cat([base_experts, expert_indices], dim=-1)
    rank = torch.cat([base_ranks, redundant_ranks], dim=-1)
    
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    logcnt.scatter_add_(1, expert_indices, torch.ones_like(expert_indices, dtype=torch.int64))
    
    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
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
        inv.scatter_(
            1,
            perm,
            torch.arange(perm.size(1), dtype=torch.int64,
                         device=perm.device).unsqueeze(0).expand(perm.shape),
        )
        return inv

    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size,
                             dtype=torch.int64,
                             device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy,
                                                num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(-1, pphy2phy)  
    pphy2mlog = (pphy2mlog.view(num_layers, num_nodes, -1) + torch.arange(
        0,
        num_logical_experts,
        num_logical_experts // num_nodes,
        device=group_pack_index.device,
    ).view(1, -1, 1)).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
    return pphy2log, pphyrank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    original_device = weight.device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight = weight.float().to(device)

    if num_groups % num_nodes == 0:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)
            
    num_layers, num_logical_experts = weight.shape
    num_redundant_experts = num_replicas - num_logical_experts
    maxlogcnt = num_redundant_experts + 1
    
    log2phy = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=device,
    )
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(num_replicas, dtype=torch.int64,
                     device=device).unsqueeze(0).expand(num_layers, -1),
    )
    
    return phy2log.to(original_device), log2phy.to(original_device), logcnt.to(original_device)
