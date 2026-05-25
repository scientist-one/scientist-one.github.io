from bisect import bisect

def count_inversions(a):
    inversions = 0
    sorted_so_far = []
    for i, u in enumerate(a):
        j = bisect(sorted_so_far, u)
        inversions += i - j
        sorted_so_far.insert(j, u)
    return inversions

def kendall_tau(ground_truth, predictions):
    total_inversions = 0
    total_2max = 0  # 2 * max_inversions (i.e. n * (n - 1))
    
    for gt, pred in zip(ground_truth, predictions):
        # gt and pred are lists of cell ids
        ranks = {cell_id: i for i, cell_id in enumerate(gt)}
        
        # Keep only cells that are in ground truth (ignoring others if any)
        pred_ranks = [ranks[x] for x in pred if x in ranks]
        
        # If lengths don't match, we might have an issue, but let's assume they do.
        n = len(gt)
        total_2max += n * (n - 1)
        total_inversions += count_inversions(pred_ranks)
        
    return 1 - 4 * total_inversions / total_2max
