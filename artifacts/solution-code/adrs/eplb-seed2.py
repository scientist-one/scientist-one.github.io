import torch

def balanced_packing_zigzag(weight: torch.Tensor, num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    num_layers, num_items = weight.shape
    items_per_pack = num_items // num_packs
    
    if items_per_pack == 1:
        pack_index = torch.arange(num_packs, dtype=torch.int64, device=weight.device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    sorted_indices = weight.sort(dim=-1, descending=True).indices
    
    row_idx = torch.arange(items_per_pack, dtype=torch.int64, device=weight.device).view(-1, 1)
    col_idx = torch.arange(num_packs, dtype=torch.int64, device=weight.device).view(1, -1)
    
    col_idx_zigzag = torch.where(row_idx % 2 == 1, num_packs - 1 - col_idx, col_idx)
    
    pack_assignment = col_idx_zigzag.reshape(1, num_items).expand(num_layers, num_items)
    rank_assignment = row_idx.expand(-1, num_packs).reshape(1, num_items).expand(num_layers, num_items)
    
    pack_index = torch.zeros_like(weight, dtype=torch.int64)
    rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
    
    pack_index.scatter_(-1, sorted_indices, pack_assignment)
    rank_in_pack.scatter_(-1, sorted_indices, rank_assignment)
    
    return pack_index, rank_in_pack


def replicate_experts(weight: torch.Tensor, num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    device = weight.device
    
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    
    if num_redundant > 0:
        arangen = torch.arange(n, dtype=torch.int64, device=device)
        current_load = weight.clone()
        for i in range(num_log, num_phy):
            redundant_indices = current_load.argmax(dim=-1)
            phy2log[:, i] = redundant_indices
            rank[:, i] = logcnt[arangen, redundant_indices]
            logcnt[arangen, redundant_indices] += 1
            current_load[arangen, redundant_indices] = weight[arangen, redundant_indices] / logcnt[arangen, redundant_indices]
            
    return phy2log, rank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    device = weight.device
    
    group_size = num_logical_experts // num_groups
    groups_per_node = num_groups // num_nodes
    phy_experts_per_gpu = num_replicas // num_gpus
    
    if num_nodes > 1 and num_groups > 1 and num_groups % num_nodes == 0:
        tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
        group_pack_index, _ = balanced_packing_zigzag(tokens_per_group, num_nodes)
        expert_home_node = group_pack_index.unsqueeze(-1).expand(-1, -1, group_size).reshape(num_layers, num_logical_experts)
    else:
        expert_home_node = torch.zeros((num_layers, num_logical_experts), dtype=torch.int64, device=device)
        
    phy2log, phyrank, logcnt = replicate_experts(weight, num_replicas)
    
    replica_home_node = expert_home_node.gather(-1, phy2log)
    log_expert_load = weight / logcnt
    replica_load = log_expert_load.gather(-1, phy2log)
    
    max_load = replica_load.max(dim=-1, keepdim=True).values + 1e-6
    sort_key = replica_home_node.double() - replica_load.double() / max_load.double()
    node_assignment_indices = sort_key.argsort(dim=-1)
    
    sorted_phy2log = phy2log.gather(-1, node_assignment_indices)
    sorted_phyrank = phyrank.gather(-1, node_assignment_indices)
    sorted_replica_load = replica_load.gather(-1, node_assignment_indices)
    
    node_local_loads = sorted_replica_load.reshape(num_layers * num_nodes, num_replicas // num_nodes)
    gpus_per_node = num_gpus // num_nodes
    pack_index, rank_in_pack = balanced_packing_zigzag(node_local_loads, gpus_per_node)
    
    pack_index = pack_index.reshape(num_layers, num_nodes, num_replicas // num_nodes)
    rank_in_pack = rank_in_pack.reshape(num_layers, num_nodes, num_replicas // num_nodes)
    node_idx = torch.arange(num_nodes, device=device).reshape(1, num_nodes, 1).expand(num_layers, num_nodes, num_replicas // num_nodes)
    
    gpu_idx = node_idx * gpus_per_node + pack_index
    physical_slot_idx = gpu_idx * phy_experts_per_gpu + rank_in_pack
    physical_slot_idx = physical_slot_idx.reshape(num_layers, num_replicas)
    
    sorted_phy2log = sorted_phy2log.reshape(num_layers, num_replicas)
    sorted_phyrank = sorted_phyrank.reshape(num_layers, num_replicas)
    
    final_phy2log = torch.zeros_like(sorted_phy2log)
    final_phyrank = torch.zeros_like(sorted_phyrank)
    
    final_phy2log.scatter_(-1, physical_slot_idx, sorted_phy2log)
    final_phyrank.scatter_(-1, physical_slot_idx, sorted_phyrank)
    
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
        final_phy2log * maxlogcnt + final_phyrank,
        torch.arange(num_replicas, dtype=torch.int64, device=device).expand(num_layers, -1),
    )
    
    return final_phy2log, log2phy, logcnt
