import torch 

def __eq(str, x,y, tol=1e-5, debug=False): 
    err = torch.abs(x-y).max()
    pass_str = "pass" if err < tol else "fail" 
    print(f"{str} : {pass_str} [err={err:0.5f}]")
    if(debug and (err > tol)):
        print(f"x\n{x}")
        print(f"y\n{y}")
        print(f"diff\n{x-y}")
        
    return err <= tol


def efficiency(flops, time):
    tflops = flops / 1e12
    time_ms = time / 1e6
    return tflops / time_ms



