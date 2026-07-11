"""MLP：模加法，暴露隐藏层激活用于 erank 计算"""
import torch
import torch.nn as nn


class ModularMLP(nn.Module):
    def __init__(self, p=97, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embed_a = nn.Embedding(p, embed_dim)
        self.embed_b = nn.Embedding(p, embed_dim)
        self.fc1 = nn.Linear(2 * embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, p)
        self.relu = nn.ReLU()

    def forward(self, a, b):
        x = torch.cat([self.embed_a(a), self.embed_b(b)], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def hidden_activation(self, a, b):
        """fc1 后隐藏层激活 [N, hidden_dim]，用于 erank 计算"""
        x = torch.cat([self.embed_a(a), self.embed_b(b)], dim=-1)
        return self.relu(self.fc1(x))
