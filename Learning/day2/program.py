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
    # dQ = torch.randn_like(Q) # for backward pass 

    print("Q Shape : " , Q.shape)
    print("K trans Shape : ", K.transpose(2,3).shape)
    MASK = torch.tril(torch.ones(seq_len,seq_len))
    print("MASK : ", MASK , "Mask Shape : ", MASK.shape)
    P = torch.matmul(Q,K.transpose(2,3)) * softmax_scale
    if causal:
        P[:, :, MASK == 0]  = float("-inf")
        print("P causal", P)
    P = torch.softmax(P.float(), dim=-1).half()
    print("Attention vals after softmax", P, "Shape of " , P.shape)
    ref_0 = torch.matmul(P,V) # ref_0 is (QKt)V
    print("Full attention operation ", ref_0[0][0], "\nshape of full attention", ref_0.shape)
print(test_op(1,4,16,2,True))
