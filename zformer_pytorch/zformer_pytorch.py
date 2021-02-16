import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from einops import repeat, rearrange
from quant_noise import quant_noise


class Zformer(nn.Module):
    def __init__(
        self,
        input_dim, 
        num_heads,
        dropout=0.0,
        scalar=700.,
        bias=True,
        q_noise=0.0,
        qn_block_size=8,
        eps=1e-9):
        super().__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.scalar = Parameter(torch.tensor(scalar))
        self.eps = eps
        self.head_dim = input_dim // num_heads
        assert (
            input_dim == self.head_dim * num_heads
        ), "input_dim must be divisible by num_heads."
        self.scaling = (input_dim // num_heads) ** -0.5

        self.q_proj = quant_noise(nn.Linear(input_dim, input_dim, bias=bias), q_noise, qn_block_size)
        self.k_proj = quant_noise(nn.Linear(input_dim, input_dim, bias=bias), q_noise, qn_block_size)
        self.v_proj = quant_noise(nn.Linear(input_dim, input_dim, bias=bias), q_noise, qn_block_size)

        self.out_proj = quant_noise(nn.Linear(input_dim, input_dim, bias=bias), q_noise, qn_block_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, input_tensor):
        input_length, batch_size, hidden_dim = input_tensor.shape
        # X = input_tensor.flatten(2)

        X = rearrange(input_tensor, 'l b d -> b l d')
        q = self.q_proj(X)
        k = self.k_proj(X)
        v = self.v_proj(X)
        
        q = q * self.scaling
        
        q = rearrange(q, 'b l (h d) -> (b h) l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> (b h) l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> (b h) l d', h=self.num_heads)
        k = rearrange(k, 'b l d -> b d l')

        X = k @ v
        X = q @ X / self.scalar
        """
        attn_probs = F.dropout(
            attn_weights,
            p=self.dropout,
            training=self.training
        )
        X = attn_probs @ v
        """
        X = rearrange(X, '(b h) l d -> b l (d h)', h=self.num_heads)
        X = self.out_proj(X)
        X = rearrange(X, 'b l d -> l b d')

        return X
