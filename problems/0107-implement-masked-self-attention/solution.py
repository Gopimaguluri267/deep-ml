import torch
import math

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return torch.matmul(X, W_q), torch.matmul(X, W_k), torch.matmul(X, W_v)

def masked_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute masked self-attention.
    """
    # Your code here
    d_k = Q.shape[-1]
    attn_wts = torch.matmul(Q, K.T)/math.sqrt(d_k)
    # if mask is not None:
    # masked_attn_wts = torch.mul(attn_wts, mask)
    masked_attn_wts = attn_wts+mask
    attn_scores = torch.softmax(masked_attn_wts, dim=-1)
    attn_out = torch.matmul(attn_scores, V)
    return attn_out
