import torch
import torch.nn as nn


class CompositionalEmbedder(nn.Module):
    def __init__(
        self, 
        embedding: nn.Embedding, 
        max_steps: int
    ):
        super().__init__()
        self.embedding = embedding
        self.max_steps = max_steps
    
    def forward(self):
        pass


class MicroStepDecoder(nn.Module):
    pass
