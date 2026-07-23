import torch
import triton
import triton.language as tl

def test_op(batch_size, num_heads, seq_len, head_dim ,casual, dtype=torch.float16):
    Q = (torch.empty(
        batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="cuda"
    ).normal_(mean=0.0,std=0.5).requires_grad_()
    )
    K = (torch.empty(
        batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="cuda"
    ).normal_(mean=0.0,std=0.5).requires_grad_()
    )
    V = (torch.empty(
        batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="cuda"
    ).normal_(mean=0.0,std=0.5).requires_grad_()
    )
    softmax_scale = 1 / (head_dim ** 0.5)
    dQ = torch.randn_like(Q)