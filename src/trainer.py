"""
训练 & 度量：erank 序参量计算
==========================
核心指标：erank = 隐藏层激活矩阵的有效秩（奇异值谱熵）
"""
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict


def compute_erank(model, a, b, n_samples=200):
    """
    隐藏层激活矩阵的有效秩。
    算法解 → 表示集中在少数方向 → erank 小
    记忆解 → 表示分散满秩 → erank 大
    """
    idx = torch.randperm(len(a), device=a.device)[:n_samples]
    with torch.no_grad():
        H = model.hidden_activation(a[idx], b[idx])
    U, S, V = torch.linalg.svd(H.float(), full_matrices=False)
    S = S[S > 1e-10]
    p = S / S.sum()
    entropy = -(p * torch.log(p + 1e-10)).sum()
    return torch.exp(entropy).item()


def train_one_run(p=97, train_frac=0.5, label_noise_ratio=0.0,
                  n_epochs=5000, lr=1e-3, weight_decay=0.5,
                  embed_dim=128, hidden_dim=256, seed=42,
                  record_interval=50, device='cpu'):
    from data import generate_data
    from models import ModularMLP

    torch.manual_seed(seed); np.random.seed(seed)

    train_a, train_b, train_y, test_a, test_b, test_y, _ = generate_data(
        p=p, train_frac=train_frac, label_noise_ratio=label_noise_ratio, seed=seed
    )
    train_a, train_b, train_y = train_a.to(device), train_b.to(device), train_y.to(device)
    test_a, test_b, test_y = test_a.to(device), test_b.to(device), test_y.to(device)

    model = ModularMLP(p=p, embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    metrics = defaultdict(list)
    batch_size = 512

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(len(train_a), device=device)
        total_loss = 0.0
        for i in range(0, len(train_a), batch_size):
            idx = perm[i:i + batch_size]
            loss = loss_fn(model(train_a[idx], train_b[idx]), train_y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        if epoch % record_interval == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                train_acc = (model(train_a, train_b).argmax(1) == train_y).float().mean().item()
                test_acc = (model(test_a, test_b).argmax(1) == test_y).float().mean().item()

            metrics['epoch'].append(epoch)
            metrics['train_loss'].append(total_loss / len(train_a))
            metrics['train_acc'].append(train_acc)
            metrics['test_acc'].append(test_acc)
            metrics['erank'].append(compute_erank(model, train_a, train_b))

    return dict(metrics), model
