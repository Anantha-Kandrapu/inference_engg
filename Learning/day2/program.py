import torch
import triton
import triton.language as tl

def test_op(batch_size, num_heads, seq_len, head_dim ,causal, dtype=torch.float16):
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
    dQ = torch.randn_like(Q) # for backward pass

    MASK = torch.tril(torch.ones(seq_len,seq_len),device="cuda")
    P = torch.matmul(Q,K.transpose(2,3)) * softmax_scale
    if causal:
        P[:, :, MASK == 0]  = float("-inf")
    P = torch.softmax(P.float(), dim=-1).half()

