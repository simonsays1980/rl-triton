"""Direct equivalence check: serial kernels vs. tree kernel, compiled scan,
compute_gae, and fp32/fp64 sequential references, incl. edge cases the timing
sweep does not cover (non-pow2 T, E not a multiple of BLOCK_E, tiny sizes,
no/all terminations, gamma=lambda=1, lambda=0). Output: equivalence_check.txt.

    python benchmarks/ablation_serial/check_equivalence.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_LOGS", "-dynamic")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(Path(__file__).resolve().parent)]
import torch
from serial_gae import gae_tree, gae_serial_env, gae_serial_time
from test_gae import reference_gae, vectorized_gae_with_truncations
from rl_triton.ops.gae import compute_gae

def ref64(r,v,d,b,g,l):
    return reference_gae(r.double(),v.double(),d.double(),g,l,bootstrap_values=b.double())

def run(E,T,term_p=0.05,g=0.99,l=0.95,seed=0,cfgs=((32,1,1,1),(64,2,8,1),(128,4,4,3),(256,8,8,1))):
    torch.manual_seed(seed)
    r=torch.randn(E,T,device="cuda"); v=torch.randn(E,T,device="cuda")
    d=(torch.rand(E,T,device="cuda")<term_p).float(); b=torch.randn(E,device="cuda")
    outs={"tree":gae_tree(r,v,d,b,g,l),"compute_gae":compute_gae(r,v,d,gamma=g,lambda_=l,last_value=b)}
    tr=torch.zeros_like(r); bs=torch.zeros_like(r); bs[:,-1]=b
    torch._dynamo.reset(); outs["scan"]=torch.compile(vectorized_gae_with_truncations)(r,v,d,tr,bs,g,l)
    rt,vt,dt=r.T.contiguous(),v.T.contiguous(),d.T.contiguous()
    for be,nw,u,ns in cfgs:
        if T%u: continue
        try:
            outs[f"env{be}/{nw}/U{u}/S{ns}"]=gae_serial_env(r,v,d,b,g,l,BLOCK_E=be,num_warps=nw,UNROLL=u,NUM_STAGES=ns)
            outs[f"time{be}/{nw}/U{u}/S{ns}"]=gae_serial_time(rt,vt,dt,b,g,l,BLOCK_E=be,num_warps=nw,UNROLL=u,NUM_STAGES=ns).T
        except Exception as e: print("   compile fail",be,nw,u,ns,type(e).__name__)
    R32=reference_gae(r,v,d,g,l,bootstrap_values=b); R64=ref64(r,v,d,b,g,l)
    ser=[k for k in outs if k.startswith(("env","time"))]
    s0=outs[ser[0]]
    worst=lambda a,B:(a.double()-B.double()).abs().max().item()
    print(f"E={E} T={T} p_term={term_p} gamma={g} lam={l}")
    print(f"  serial vs fp32 ref  : max {max(worst(outs[k],R32) for k in ser):.2e}  bit-identical to fp32 ref: {all(torch.equal(outs[k],R32) for k in ser)}")
    print(f"  serial vs fp64 ref  : max {max(worst(outs[k],R64) for k in ser):.2e}   tree vs fp64: {worst(outs['tree'],R64):.2e}  scan vs fp64: {worst(outs['scan'],R64):.2e}")
    print(f"  serial vs tree      : max {max(worst(outs[k],outs['tree']) for k in ser):.2e}  bit-identical: {all(torch.equal(outs[k],outs['tree']) for k in ser)}")
    print(f"  serial vs scan      : max {max(worst(outs[k],outs['scan']) for k in ser):.2e}  bit-identical: {all(torch.equal(outs[k],outs['scan']) for k in ser)}")
    print(f"  serial configs agree bitwise with each other: {all(torch.equal(outs[k],s0) for k in ser)}  ({len(ser)} variants)")
    print(f"  tree == compute_gae bitwise: {torch.equal(outs['tree'],outs['compute_gae'])}")
    ok=all(torch.allclose(outs[k],R32,atol=1e-4,rtol=1e-4) for k in outs)
    print(f"  ALL within 1e-4 of reference: {ok}")

run(16384,128); run(512,4096); run(32768,8)
run(100,40)                     # E not multiple of BLOCK_E, T not power of 2
run(1,8); run(3,1,cfgs=((32,1,1,1),))   # degenerate sizes
run(1024,256,term_p=0.0); run(1024,256,term_p=1.0)   # no / all terminations
run(1024,256,g=1.0,l=1.0); run(1024,256,g=0.999,l=0.0)
