import torch
tile = 16
#####
# Helpers
#   
def __eq(str, x,y, tol=1e-5, debug=False): 
    err = torch.abs(x-y).max()
    pass_str = "pass" if err < tol else "fail" 
    print(f"{str} : {pass_str} [err={err:0.5f}]")
    if(debug and (err > tol)):
        print(f"x\n{x}")
        print(f"y\n{y}")
        print(f"diff\n{x-y}")
        
    return err <= tol

def _rtile(b,n,d,dt): return torch.randn(b,n,d,device='cuda', dtype=dt)/(n*d)
def _rhtile(b,h,n,d,dt): return torch.randn(b,h,n,d,device='cuda', dtype=dt)/(n*d)
def _rones(b,n,d,dt): return torch.ones(b,n,d,device='cuda', dtype=dt)

def print_tiles(str, t):
    for i in range(t.size(0)):
        for j in range(t.size(1)//tile):
            print(f"{str} TILE batch={i} tile={j}")
            print(f"{t[i,j*tile:(j+1)*tile,:]}")