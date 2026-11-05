"""Synthetic algebra/shape checks only; no optimizer step or scientific result."""
from unittest.mock import patch
import numpy as np
import torch
from common import ROOT,write_json,now,protocol
from learning import Score3D
from diffusion import CapturedHeun

torch.manual_seed(8237)
g=np.load(ROOT/'checks/n40_shear0.050_relaxed.npz')
ids=np.flatnonzero(np.all((g['sites']>=-8)&(g['sites']<8),axis=1))
qref=torch.tensor(g['qref'][ids].ravel(),dtype=torch.float32)
P,_=torch.linalg.qr(torch.randn(len(qref),16))
s=torch.linspace(.1,2,16)
parameters=dict(P=P,sqrt_diagonal=torch.full_like(qref,.005),expansion=torch.sqrt(1+s*s)-1,
    contraction=1/torch.sqrt(1+s*s)-1,means=torch.stack([qref]*len(protocol()['temperatures_K']['train'])),temperatures=torch.tensor(protocol()['temperatures_K']['train'],dtype=torch.float32),
    qref=qref,sites=torch.tensor(g['sites'][ids]))
with patch('learning.torch.load',return_value=parameters):model=Score3D().cuda()
z=torch.randn(2,len(qref),device='cuda',requires_grad=True);T=torch.tensor([100.,400.],device='cuda')
x=model.wh.physical(z,T)
roundtrip=float(abs(model.wh.whiten(x,T)-z).max())
f=torch.randn_like(x)
gradient=torch.autograd.grad((f*x).sum(),z)[0]
chain=float(abs(gradient-model.wh.force_to_z(f,T)).max())
pred=model(z.detach(),torch.tensor([.25,.75],device='cuda'),T)
assert pred.shape==z.shape and bool(torch.isfinite(pred).all())
initial=float(pred.abs().max())
loss=(pred-torch.randn_like(pred)).square().mean();loss.backward()
gradients=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
model.eval()
with torch.no_grad():
 graph=CapturedHeun(model,300,5,2,False)
 noise=torch.randn(2,len(qref),device='cuda')
 identity=float(abs(graph.sample(noise).cuda()-noise).max())
result=dict(at=now(),scope='synthetic algebra and CUDA graph check; no training updates',
 roundtrip_max_error=roundtrip,force_chain_rule_max_error=chain,initial_residual_max=initial,
 finite_gradients=gradients,zero_residual_Heun_identity_error=identity,
 passed=roundtrip<1e-3 and chain<1e-5 and initial==0 and gradients and identity==0)
write_json(ROOT/'checks/learning_implementation.json',result)
print(result,flush=True)
assert result['passed']
