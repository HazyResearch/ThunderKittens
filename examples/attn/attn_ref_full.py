import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import opt_einsum as oe


d_model = 32
l_max = 8
batch_size = 2
softmax_scale = 1 / math.sqrt(d_model)
use_denom = True


class SelfAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.l_max = l_max
        self.num_heads = 2
        self.num_key_value_heads = 2
        self.head_dim = self.d_model // self.num_key_value_heads

        self.proj_q = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=False)
        self.proj_k = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=False)
        self.proj_v = nn.Linear(self.d_model, self.num_key_value_heads * self.head_dim, bias=False)
        self.proj_o = nn.Linear(self.num_heads * self.head_dim, self.d_model, bias=False)
        self.dropout = nn.Identity()

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        b, l, _ = hidden_states.size()
        q = self.proj_q(hidden_states)
        k = self.proj_k(hidden_states)
        v = self.proj_v(hidden_states)
        
        q = q.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, l, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, l, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        A0 = torch.einsum("bhnd,bhmd->bhnm",q, k) * softmax_scale
        numerator  = torch.exp(A0) 
        denominator = torch.exp(A0).sum(dim=-1, keepdim=True)
        if use_denom:
            A = numerator / denominator
        else:
            A = numerator
        attn_output  = torch.einsum("bhnm,bhmd->bhnd",A,v)

        attn_output = attn_output.reshape(b, l, self.d_model)
        attn_output = self.proj_o(attn_output)
        return attn_output


class SelfAttnFunction(torch.autograd.Function):
    """
    We can implement custom autograd by subclassing torch.autograd.Function.
    """

    @staticmethod
    def forward(ctx, Q, K, V):
        """
        ctx is a context to save info for backward, using ctx.save_for_backward
        """

        A0 = torch.einsum("bhnd,bhmd->bhnm",Q, K) * softmax_scale
        numerator  = torch.exp(A0) 
        denominator = torch.exp(A0).sum(dim=-1, keepdim=True)
        if use_denom:
            A = numerator / denominator
        else:
            A = numerator
        y  = torch.einsum("bhnm,bhmd->bhnd",A,V)
        ctx.save_for_backward(Q, K, V, A, A0, numerator, denominator)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, A, A0, numerator, denominator = ctx.saved_tensors
        
        running_Q_grad = torch.zeros_like(Q)
        running_K_grad = torch.zeros_like(K)
        running_V_grad = torch.zeros_like(V)

        # Gradient of output (y) with respect to V
        running_V_grad += torch.einsum("bhnm,bhnd->bhmd", A, grad_output)

        # Gradient of output (y) with respect to A
        dL_dA = torch.einsum("bhnd,bhmd->bhnm", grad_output, V)

        if use_denom:
            # Recall A is the softmax result
            # Recall the derivative is a piece-wise function
            # * dL_dA * A is the elementwise product of the gradient of the loss w/r/t the softmax output and the softmax output itself (i=k case)
            # * dl_dA * A.sum(dim=-1, keepdim=True) is the i != k case
            dL_dA0 = A * (dL_dA - (dL_dA * A).sum(dim=-1, keepdim=True))
        else:
            dL_dA0 = A * dL_dA

        dL_dA0 *= softmax_scale 

        # Gradient of A0 with respect to Q and K
        running_Q_grad += torch.einsum("bhnm,bhmd->bhnd", dL_dA0, K)
        running_K_grad += torch.einsum("bhnm,bhnd->bhmd", dL_dA0, Q)

        return running_Q_grad, running_K_grad, running_V_grad, None, None


class SelfAttnManual(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.l_max = l_max
        self.num_heads = 2
        self.num_key_value_heads = 2
        self.head_dim = self.d_model // self.num_key_value_heads
        self.eps = 1e-12

        self.feature_map = SelfAttnFunction
        self.proj_q = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=False)
        self.proj_k = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=False)
        self.proj_v = nn.Linear(self.d_model, self.num_key_value_heads * self.head_dim, bias=False)
        self.proj_o = nn.Linear(self.num_heads * self.head_dim, self.d_model, bias=False)
        self.dropout = nn.Identity()

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        b, l, _ = hidden_states.size()
        q = self.proj_q(hidden_states)
        k = self.proj_k(hidden_states)
        v = self.proj_v(hidden_states)
        
        q = q.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, l, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, l, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        attn_output = self.feature_map.apply(q, k, v)
        attn_output = attn_output.reshape(b, l, self.d_model)
        y = self.proj_o(attn_output)
        y = self.dropout(y)
        return y.to(hidden_states.dtype)


if __name__ == "__main__":
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    seq_mixer = SelfAttn()
    seq_mixer_manual = SelfAttnManual()

    # set the weights to be the same
    seq_mixer.proj_q.weight = torch.nn.Parameter(seq_mixer_manual.proj_q.weight.clone())
    seq_mixer.proj_k.weight = torch.nn.Parameter(seq_mixer_manual.proj_k.weight.clone())
    seq_mixer.proj_v.weight = torch.nn.Parameter(seq_mixer_manual.proj_v.weight.clone())
    seq_mixer.proj_o.weight = torch.nn.Parameter(seq_mixer_manual.proj_o.weight.clone())

    # input tensor
    x = torch.randn(batch_size, l_max, d_model)
    y = seq_mixer(x)
    print()
    y_manual = seq_mixer_manual(x)
    print()
    print(f"{y.shape=}")
    print(f"{y_manual.shape=}")

    # check that the outputs are the same from forward pass
    print(f"\nForward pass:")
    print(torch.norm(y - y_manual))

    # # check that the backwards pass is the same
    print(f"\nBackward pass:")
    y.retain_grad()
    y.sum().backward()
    y_manual.sum().backward()

    # compare the gradients
    print(f"\nGradient max:")
    try:
        print("proj_q: ",torch.max(seq_mixer.proj_q.weight.grad - seq_mixer_manual.proj_q.weight.grad),)
    except:
        print(f"Skipping q grad check.")
    try:
        print("proj_k: ",torch.max(seq_mixer.proj_k.weight.grad - seq_mixer_manual.proj_k.weight.grad),)
    except:
        print(f"Skipping k grad check.")
    try:
        print("proj_v: ",torch.max(seq_mixer.proj_v.weight.grad - seq_mixer_manual.proj_v.weight.grad),)
    except:
        print(f"Skipping v grad check.")
    print("proj_o: ",torch.max(seq_mixer.proj_o.weight.grad - seq_mixer_manual.proj_o.weight.grad),)