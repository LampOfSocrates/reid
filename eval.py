import numpy as np
import torch

def euclidean_distance(qf, gf):
    """
    Computes pairwise Euclidean distance exactly the way standard Re-ID metric asks.
    qf: query features (num_queries, feature_dim)
    gf: gallery features (num_gallery, feature_dim)
    """
    m = qf.shape[0]
    n = gf.shape[0]
    # dist_mat = qf^2 + gf^2 - 2 * qf * gf
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def calculate_rank1_map(distmat, q_pids, g_pids, q_camids, g_camids):
    """
    Standard ReID metric calculation dropping same identity + same camera matches.
    """
    num_q, num_g = distmat.shape
    
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    all_cmc = []
    all_AP = []
    
    for q_idx in range(num_q):
        # Remove exactly same ID + same Camera from gallery before calculation
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        
        # Valid galleries are those NOT (same PID and same CAM)
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        
        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            # this condition prevents crashing when there are no valid positive samples
            continue
            
        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1 # We just care about whether it appears at rank k
        all_cmc.append(cmc[:50]) # store cmc till rank-50
        
        # Calculate AP
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / len(all_cmc)
    rank1 = all_cmc[0]
    mAP = np.mean(all_AP)
    return rank1, mAP
