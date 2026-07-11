"""数据生成：模加法 + 可控 label_noise_ratio"""
import numpy as np
import torch


def generate_data(p=97, train_frac=0.5, label_noise_ratio=0.0, seed=42):
    """
    p: 模数
    label_noise_ratio: 训练标签被随机篡改的比例 (0.0-1.0)
    测试集始终用全部正确标签。
    """
    rng = np.random.RandomState(seed)
    all_pairs = [(a, b) for a in range(p) for b in range(p)]
    n_train = int(len(all_pairs) * train_frac)
    idx = rng.permutation(len(all_pairs))
    train_pairs = [all_pairs[i] for i in idx[:n_train]]

    train_a = np.array([a for a, b in train_pairs], dtype=np.int64)
    train_b = np.array([b for a, b in train_pairs], dtype=np.int64)
    train_y = np.array([(a + b) % p for a, b in train_pairs], dtype=np.int64)

    if label_noise_ratio > 0:
        n_noise = int(len(train_y) * label_noise_ratio)
        noise_idx = rng.choice(len(train_y), n_noise, replace=False)
        for i in noise_idx:
            correct = train_y[i]
            wrong = correct
            while wrong == correct:
                wrong = rng.randint(0, p)
            train_y[i] = wrong

    test_a = np.array([a for a, b in all_pairs], dtype=np.int64)
    test_b = np.array([b for a, b in all_pairs], dtype=np.int64)
    test_y = np.array([(a + b) % p for a, b in all_pairs], dtype=np.int64)

    return (
        torch.tensor(train_a, dtype=torch.long),
        torch.tensor(train_b, dtype=torch.long),
        torch.tensor(train_y, dtype=torch.long),
        torch.tensor(test_a, dtype=torch.long),
        torch.tensor(test_b, dtype=torch.long),
        torch.tensor(test_y, dtype=torch.long),
        p
    )
