import torch
torch.random.manual_seed(42)
from tqdm import trange
import sys

D_QK = 128
D_VO = 128

def generate_inputs(testname, B, H, N):
    alphas = torch.rand((H,), dtype=torch.float32, device='cuda')*3
    betas = torch.rand((H,), dtype=torch.float32, device='cuda')
    if testname == 'randn':
        q = (torch.randn((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda'))
        k = (torch.randn((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda'))
        v = (torch.randn((B, H, N, D_VO), dtype=torch.bfloat16, device='cuda')) / 5
        qmap = (torch.randn((H, D_QK, 64), dtype=torch.bfloat16, device='cuda'))
        kmap = (torch.randn((H, D_QK, 64), dtype=torch.bfloat16, device='cuda'))
    elif testname == 'ones':
        q = (torch.ones((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda')) / 5
        k = (torch.ones((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda')) / 5
        v = (torch.ones((B, H, N, D_VO), dtype=torch.bfloat16, device='cuda')) / 5
        qmap = (torch.ones((H, D_QK, 64), dtype=torch.bfloat16, device='cuda')) / 20
        kmap = (torch.ones((H, D_QK, 64), dtype=torch.bfloat16, device='cuda')) / 20
    elif testname == 'qk_test': # this test breaks if sequence length is set <128
        q = torch.eye(D_QK).reshape((1,1,D_QK,D_QK)).repeat(B, H, N//D_QK, 1)*10
        q = q.to(dtype=torch.bfloat16, device='cuda')
        k = torch.eye(D_QK).reshape((1,1,D_QK,D_QK)).repeat(B, H, N//D_QK, 1)*10
        k = k.to(dtype=torch.bfloat16, device='cuda')
        v = torch.eye(D_VO).reshape((1,1,D_VO,D_VO)).repeat(B, H, N//D_VO, 1)*10
        v = v.to(dtype=torch.bfloat16, device='cuda')
        qmap = (torch.ones((H, D_QK, 64), dtype=torch.bfloat16, device='cuda')) / 20
        kmap = (torch.ones((H, D_QK, 64), dtype=torch.bfloat16, device='cuda')) / 20
    elif testname == 'v_or':
        q = (torch.randn((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda')) / 5
        k = (torch.randn((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda')) / 5
        v = (torch.arange(D_VO*2, dtype=torch.bfloat16, device='cuda')/D_VO).reshape((1,1,2,-1)).repeat(B, H, N//2, 1) / 5
        qmap = (torch.ones((H, D_QK, 64), dtype=torch.bfloat16, device='cuda')) / 20
        kmap = (torch.ones((H, D_QK, 64), dtype=torch.bfloat16, device='cuda')) / 20
    elif testname == 'dbg':
        q = (torch.ones((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda')) / 5
        k = (torch.ones((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda')) / 5
        v = (torch.ones((B, H, N, D_VO), dtype=torch.bfloat16, device='cuda')) / 10
        qmap = (torch.eye(64, dtype=torch.bfloat16, device='cuda')).repeat((H,2,1))
        kmap = (torch.eye(64, dtype=torch.bfloat16, device='cuda')).repeat((H,2,1))
    else:
        print('bad testname')
        sys.exit(0)
    return q, k, v, qmap, kmap, alphas, betas

def pytorch_softmax_gt(x, map_mat, label=None):

    x = torch.einsum('bhmd,hdn->bhmn', x, map_mat)

    if label is not None:
        with open(label, 'w') as f:
            for row in x[0,0,0:256]:
                f.write(' '.join(map(lambda k: f"{k:8.4f}", row[:256].tolist())) + '\n')
    
    x_pos = x
    x_neg = -x
    
    x_pos_max = torch.amax(x_pos, dim=-1, keepdim=True)
    x_neg_max = torch.amax(x_neg, dim=-1, keepdim=True)
    
    x_pos = x_pos - x_pos_max
    x_neg = x_neg - x_neg_max
    
    x_pos_num = torch.exp(x_pos)
    x_pos_den = torch.sum(torch.exp(x_pos), dim=-1, keepdim=True)
    
    x_neg_num = torch.exp(x_neg)
    x_neg_den = torch.sum(torch.exp(x_neg), dim=-1, keepdim=True)
    
    x_pos = x_pos_num / x_pos_den
    x_neg = x_neg_num / x_neg_den
    
    x = torch.cat([x_pos, x_neg], dim=-1).clamp(min=1e-6)
    
    return x

generator_mat = torch.block_diag(*[torch.ones((64,64), device='cuda')]*256) # 16384 x 16384 should be big enough for most porpoises
generator_mat += torch.roll(generator_mat, -64, -1) # this adds the terracing
lin_mask = torch.tril(1-generator_mat).reshape((1,1,16384,16384))
exp_mask = torch.tril(generator_mat).reshape((1,1,16384,16384))
exp_mask = 10000*exp_mask - 10000 # we actually want to effectively subtract infinity instead with this mask, since it should happen pre-exp

def pytorch_test(Q, K, V, Qmap, Kmap, alphas, betas):
    
    Qs = pytorch_softmax_gt(Q, Qmap)
    Ks = pytorch_softmax_gt(K, Kmap)
    
    a_lin = torch.einsum('bhmd,bhnd->bhmn', Qs, Ks).to(torch.float32)
    a_exp = torch.einsum('bhmd,bhnd->bhmn', Q, K).to(torch.float32)
    # mask
    a_lin *= lin_mask[:,:,:a_lin.shape[2], :a_lin.shape[3]] * alphas.reshape((1,-1,1,1)) # zero out unwanted entries
    a_exp += exp_mask[:,:,:a_exp.shape[2], :a_exp.shape[3]] # subtract infinity off of unwanted entries
    a_exp -= a_exp.amax(dim=-1, keepdim=True)
    a_exp = torch.exp(a_exp / (128**.5)) * betas.reshape((1,-1,1,1))

    a = a_exp + a_lin
    a = (a / (a.sum(dim=-1, keepdim=True)+1e-6)).to(torch.bfloat16) # normalize
    
    out = torch.einsum('bhmn,bhnd->bhmd', a, V).to(torch.bfloat16)
    
    kv_state = torch.einsum('bhlf,bhld->bhfd', Ks[:,:,:-64,:], V[:,:,:-64,:]).to(torch.float32).detach()
    k_state  = Ks[:,:,:-64,:].to(torch.float32).sum(dim=-2).detach()    
    return out, kv_state, k_state

NAMES = ['randn', 'ones', 'qk_test', 'v_or', 'dbg']
TESTNAME = sys.argv[1]

# get inputs
B, H, N = 2, 2, 1024
assert TESTNAME in NAMES, f"bad testname {TESTNAME}"
q, k ,v, qmap, kmap, alphas, betas = generate_inputs(TESTNAME, B, H, N)
out, kv_state, k_state = pytorch_test(q, k, v, qmap, kmap, alphas, betas)

with open(f'{TESTNAME}.txt', 'w') as f:
    qf = q.to(torch.float32).flatten().cpu().numpy().tolist()
    kf = k.to(torch.float32).flatten().cpu().numpy().tolist()
    vf = v.to(torch.float32).flatten().cpu().numpy().tolist()
    qmf = qmap.to(torch.float32).flatten().cpu().numpy().tolist()
    kmf = kmap.to(torch.float32).flatten().cpu().numpy().tolist()
    af = alphas.to(torch.float32).cpu().numpy().tolist()
    bf = betas.to(torch.float32).cpu().numpy().tolist()
    of = out.to(torch.float32).flatten().cpu().numpy().tolist()
    kvf = kv_state.flatten().cpu().numpy().tolist()
    ksf = k_state.flatten().cpu().numpy().tolist()

    # 1. alpha and beta 
    for i in trange(len(af)):
        f.write(repr(af[i]))
        f.write(' ')

    for i in trange(len(bf)):
        f.write(repr(bf[i]))
        f.write(' ')

    # 2. qmap and kmap
    for i in trange(len(qmf)):
        f.write(repr(qmf[i]))
        f.write(' ')

    for i in trange(len(kmf)):
        f.write(repr(kmf[i]))
        f.write(' ')

    # 3. q, k, v, out
    for i in trange(len(qf)):
        f.write(repr(qf[i]))
        f.write(' ')
    
    for i in trange(len(kf)):
        f.write(repr(kf[i]))
        f.write(' ')

    for i in trange(len(vf)):
        f.write(repr(vf[i]))
        f.write(' ')
    
    for i in trange(len(of)):
        f.write(repr(of[i]))
        f.write(' ')
    
    # k then kv state
    for i in trange(len(ksf)):
        f.write(repr(ksf[i]))
        f.write(' ')

    for i in trange(len(kvf)):
        f.write(repr(kvf[i]))
        f.write(' ')
