# utils/clustering.py （也可以直接写在 train_ant.py 顶部）

import numpy as np
import torch
from sklearn.cluster import KMeans


def compute_medoids_from_buffer(
    replay_buffer,
    device,
    sample_size=5000,
    num_clusters=64,
):
    """
    从 replay_buffer 中随机采样 sample_size 个状态，对这些状态做 KMeans 聚类，
    然后在每个簇内选出距离质心最近的样本作为 medoid（真实状态）。

    返回:
        medoids_torch: [num_clusters, state_dim] 的 torch.FloatTensor, 在指定 device 上。
    """
    # 1) 从 buffer 里拿样本
    states = replay_buffer.random_state_batch(sample_size)   # [N, Ds], numpy 或 list
    states_np = np.asarray(states)                           # [N, Ds]
    N, Ds = states_np.shape

    # 2) KMeans 聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(states_np)
    centers = kmeans.cluster_centers_   # [K, Ds]
    labels = kmeans.labels_            # [N]

    # 3) 对每个簇，找距离中心最近的样本 index
    medoid_indices = []
    for k in range(num_clusters):
        cluster_idx = np.where(labels == k)[0]
        if cluster_idx.size == 0:
            # 可能某些簇是空的，退化到随机一个样本
            medoid_indices.append(np.random.randint(0, N))
            continue
        cluster_points = states_np[cluster_idx]             # [Nk, Ds]
        diffs = cluster_points - centers[k][None, :]        # [Nk, Ds]
        dists = np.sum(diffs ** 2, axis=1)                  # [Nk]
        best_local = cluster_idx[np.argmin(dists)]
        medoid_indices.append(best_local)

    medoids_np = states_np[medoid_indices]  # [K, Ds]
    medoids_torch = torch.as_tensor(medoids_np, dtype=torch.float32, device=device)
    return medoids_torch



@torch.no_grad()
def farthest_point_sampling(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS) / k-center greedy.
    选出来的点一定是“真实样本点”（满足你对 medoids 的要求：不能是均值）。

    Args:
        x: [N, D] float tensor
        k: number of points to pick

    Returns:
        idx: [k] long tensor indices into x
    """
    device = x.device
    N = x.size(0)
    if k >= N:
        return torch.arange(N, device=device, dtype=torch.long)

    # 随机选第一个点
    first = torch.randint(low=0, high=N, size=(1,), device=device).item()
    idx = torch.empty((k,), device=device, dtype=torch.long)
    idx[0] = first

    # dist[n] = 当前 n 点到已选集合的最小距离（平方）
    dist = ((x - x[first]) ** 2).sum(dim=-1)  # [N]

    for i in range(1, k):
        farthest = torch.argmax(dist).item()
        idx[i] = farthest
        new_dist = ((x - x[farthest]) ** 2).sum(dim=-1)
        dist = torch.minimum(dist, new_dist)

    return idx


@torch.no_grad()
def compute_medoids_from_buffer(
    replay_buffer,
    device,
    sample_size: int = 5000,
    num_clusters: int = 64,
    use_dims=(0, 1),
):
    """
    从 replay buffer 采样 sample_size 个 state，用 FPS 选 num_clusters 个“medoids”（真实 state）。

    Args:
        replay_buffer: 需要实现 random_state_batch(batch_size)
        device: torch.device or str
        sample_size: 从 buffer 抽样数
        num_clusters: medoids 数
        use_dims: 距离计算使用的维度（例如 (0,1) 表示 xy）
                 注意：返回的 medoids 仍是完整 state_dim

    Returns:
        medoids: [num_clusters, state_dim] float tensor on device
    """
    # buffer 可能还没数据
    if getattr(replay_buffer, "_size", 0) <= 0:
        raise RuntimeError("Replay buffer is empty, cannot compute medoids.")

    # 抽样
    sample_size = int(sample_size)
    states_np = replay_buffer.random_state_batch(sample_size)  # [N, Ds] np
    states = torch.as_tensor(states_np, dtype=torch.float32, device=device)

    N = states.size(0)
    if N <= num_clusters:
        return states[:num_clusters]

    # 距离计算子空间
    if isinstance(use_dims, (tuple, list)):
        x = states[:, list(use_dims)]
    else:
        x = states[:, use_dims]

    idx = farthest_point_sampling(x, num_clusters)
    medoids = states[idx]  # 真实点

    return medoids
