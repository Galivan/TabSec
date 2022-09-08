import torch.nn as nn
import torch
from functools import reduce



class EmbeddingMul(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, eps=1e-4):
        super(EmbeddingMul, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.indices = torch.arange(num_embeddings, requires_grad=False)
        self.weights = torch.randn((num_embeddings, embedding_dim), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        a = (x.unsqueeze(1)**2)
        b = self.indices.to(x.device)**2
        c = (a - b)+self.eps
        d = 1/c
        d = d*self.eps
        e = d @ self.weights.to(x.device)
        return e

if __name__ == "__main__":
    layer = EmbeddingMul(10, 5)
    x = torch.tensor([1.,2.,3.,1.,2.,3.], requires_grad=True)
    y = layer(x)
    print(y)