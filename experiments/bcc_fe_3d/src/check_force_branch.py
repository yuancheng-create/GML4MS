"""Tiny synthetic optimizer fixture: verify the actual force-to-DSM training path."""
import copy
import json
import math
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
import learning
from common import ROOT,now,protocol,sha256,write_json


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__();self.output=nn.Linear(8,8)
        self.config=dict(dimension=8,architecture='synthetic_optimizer_fixture')
    def forward(self,z,t,T):return self.output(z)


def main():
    fixture=Path(tempfile.mkdtemp(prefix='bcc_fe_force_branch_',dir='/tmp'))
    out=fixture/'data/processed';out.mkdir(parents=True)
    (fixture/'models').mkdir();(fixture/'results').mkdir()
    (out/'preconditioner.pt').write_bytes(b'synthetic identity only')
    rng=np.random.default_rng(814)
    temperatures=protocol()['temperatures_K']['train']
    for role in ['train','validation']:
        for name in ['z','residual']:np.save(out/f'{role}_{name}.npy',rng.normal(size=(20*len(temperatures),8)).astype('float32'))
        np.save(out/f'{role}_T.npy',np.repeat(temperatures,20).astype('float32'))
    p=copy.deepcopy(protocol());p['models']['updates_per_model_max']=4
    p['models']['checkpoint_steps']=[1,2,3,4];p['models']['warm_start']['early_window_max_step']=2
    with patch.multiple(learning,ROOT=fixture,protocol=lambda:p,Score3D=TinyModel):
        learning.train('force')
        parent=fixture/'models/force/resume_step_000002.pt'
        write_json(fixture/'results/switch_selection.json',dict(selected_step=2,
            validation_temperatures=p['temperatures_K']['train'],parent_resume_checkpoint=str(parent.relative_to(fixture)),
            parent_resume_sha256=sha256(parent)))
        learning.train('force_then_denoising')
        learning.train('denoising')
    state=torch.load(parent,map_location='cuda',weights_only=False)
    model=TinyModel().cuda();ema=copy.deepcopy(model).eval()
    model.load_state_dict(state['model']);ema.load_state_dict(state['ema'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5);optimizer.load_state_dict(state['optimizer'])
    generator=torch.Generator(device='cuda');generator.set_state(state['rng'].cpu())
    z=torch.from_numpy(np.load(out/'train_z.npy')).cuda();T=torch.from_numpy(np.load(out/'train_T.npy')).cuda()
    for step in [3,4]:
        ix=torch.randint(len(z),(64,),device='cuda',generator=generator)
        tau=torch.rand(64,device='cuda',generator=generator)
        eps=torch.randn((64,8),device='cuda',generator=generator)
        a=torch.cos(math.pi*tau/2)[:,None];s=torch.sin(math.pi*tau/2)[:,None];noisy=a*z[ix]+s*eps
        optimizer.zero_grad(set_to_none=True)
        # Independent direct expression for the unscaled denoising objective.
        loss=(s*noisy+a*s*model(noisy,tau,T[ix])-eps).square().mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True)
        for group in optimizer.param_groups:group['lr']=2e-4*(.1+.9*(1+math.cos(math.pi*step/4))/2)
        optimizer.step()
        with torch.no_grad():
            for average,current in zip(ema.parameters(),model.parameters()):average.lerp_(current,.001)
    actual=torch.load(fixture/'models/force_then_denoising/last.pt',map_location='cuda',weights_only=False)
    assert all(torch.equal(v,actual['model'][k]) for k,v in model.state_dict().items())
    assert all(torch.equal(v,actual['ema'][k]) for k,v in ema.state_dict().items())
    assert torch.equal(generator.get_state().cpu(),actual['rng'].cpu())
    assert [r['training_phase'] for r in actual['history']]==['force','force','denoising','denoising']
    assert (fixture/'models/force_then_denoising/step_000002.pt').resolve()==(fixture/'models/force/step_000002.pt').resolve()
    ends=[json.loads((fixture/'models'/m/'complete.json').read_text()) for m in p['models']['all_objectives']]
    assert len({x['final_training_rng_sha256'] for x in ends})==1
    write_json(ROOT/'checks/force_branch_implementation.json',dict(passed=True,at=now(),
        learning_sha256=sha256(ROOT/'src/learning.py'),fixture_directory=str(fixture),
        raw_model_and_EMA_match_independent_unscaled_DSM_replay=True,
        optimizer_and_RNG_inherited=True,shared_prefix_symlink_verified=True,matched_final_rng=True,
        scope='Four updates of an 8-dimensional synthetic linear network in a temporary directory; not an additional scientific model or MD experiment.'))
    print('Force-to-denoising branch exactly matches independent optimizer replay.',flush=True)


if __name__=='__main__':main()
