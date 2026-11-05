"""Three matched 3D score models; no test-temperature data used in fitting."""
import copy
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from common import ROOT,KB,protocol,write_json,now,sha256

LEARNING_SOURCE_SHA256=sha256(Path(__file__))


def tensor_hash(items):
    digest=hashlib.sha256()
    for name,value in items:
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def paths_for(role,T):
    paths=[]
    for p in sorted((ROOT/'data').glob(f'{role}_n*_T{T}_s*/meta.json')):
        d=json.loads(p.read_text())
        if d.get('complete'):paths.append(p.parent)
    return paths


def concatenate(paths,name):
    return np.concatenate([np.load(p/f'{name}.npy',mmap_mode='r') for p in paths])


def preprocess(rank=2048):
    out=ROOT/'data/processed';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():return
    p=protocol();temperatures=p['temperatures_K']['train']
    means=[];scaled=[];source={};torch.manual_seed(620831)
    geometry=None
    for T in temperatures:
        paths=paths_for('train',T)
        if len(paths)!=4:raise RuntimeError(f'Expected 4 independent training chains at {T} K.')
        x=concatenate(paths,'coordinates').reshape(-1,24381)
        mu=x.mean(axis=0,dtype=np.float64).astype(np.float32);means.append(mu)
        scaled.append(torch.from_numpy((x-mu)/math.sqrt(T)).cuda())
        source[str(T)]=[str(x.relative_to(ROOT)) for x in paths]
        geometry=dict(np.load(paths[0]/'geometry.npz'))
    X=torch.cat(scaled);del scaled
    rank=min(rank,min(X.shape)-1)
    U,S,V=torch.pca_lowrank(X,q=rank,center=False,niter=4)
    eigen=S.square()/len(X)
    variance=X.square().mean(0)
    # Quantify approximation of the *preconditioner*, not of the target density.
    probes=torch.randn(X.shape[1],32,device='cuda')
    truth=X.T@(X@probes)/len(X)
    validation_truth=torch.zeros_like(truth);validation_count=0;validation_sources={}
    for T in temperatures:
        paths=paths_for('validation',T)
        validation_sources[str(T)]=[str(x.relative_to(ROOT)) for x in paths]
        y=concatenate(paths,'coordinates').reshape(-1,24381)
        y=(y-y.mean(0,dtype=np.float64).astype(np.float32))/math.sqrt(T)
        for begin in range(0,len(y),256):
            block=torch.tensor(y[begin:begin+256],device='cuda')
            validation_truth+=block.T@(block@probes);validation_count+=len(block)
        del y,block
    validation_truth/=validation_count
    audits=[]
    for k in sorted(set([min(512,rank),min(1024,rank),rank])):
        retained=(V[:,:k].square()*eigen[:k]).sum(1)
        diag=(variance-retained).clamp_min(float(variance.mean())*1e-4)
        approximation=diag[:,None]*probes+V[:,:k]@(eigen[:k,None]*(V[:,:k].T@probes))
        audits.append(dict(rank=k,covariance_probe_relative_error=float(torch.linalg.norm(approximation-truth)/torch.linalg.norm(truth)),
                           independent_validation_probe_relative_error=float(torch.linalg.norm(approximation-validation_truth)/torch.linalg.norm(validation_truth)),
                           leading_variance_fraction=float(eigen[:k].sum()/variance.sum())))
    diagonal=(variance-(V.square()*eigen).sum(1)).clamp_min(float(variance.mean())*1e-4)
    # C=D+V diag(eigen) V^T = D^(1/2) (I+P s^2 P^T) D^(1/2).
    scaled_factor=V*eigen.sqrt()[None,:]/diagonal.sqrt()[:,None]
    P,s,_=torch.linalg.svd(scaled_factor,full_matrices=False)
    params=dict(P=P.cpu(),sqrt_diagonal=diagonal.sqrt().cpu(),
                expansion=(torch.sqrt(1+s*s)-1).cpu(),
                contraction=(1/torch.sqrt(1+s*s)-1).cpu(),
                means=torch.tensor(np.array(means)),temperatures=torch.tensor(temperatures,dtype=torch.float32),
                qref=torch.tensor(geometry['qref'].reshape(-1),dtype=torch.float32),
                sites=torch.tensor(geometry['sites']),rank=rank,
                full_atoms=int(geometry['full_atoms']))
    torch.save(params,out/'preconditioner.pt')
    write_json(out/'covariance_audit.json',dict(at=now(),rank_audits=audits,chosen_rank=rank,
        interpretation='Approximation used only for an invertible shared preconditioner. The Gaussian comparison samples the full empirical correlated covariance separately. Target and force identities are unchanged.',
        training_sources=source,independent_audit_sources=validation_sources,test_temperature_data_used=False))
    del X,U,S,V,P,s,scaled_factor,truth,validation_truth,probes,approximation
    torch.cuda.empty_cache()
    wh=Preconditioner().cuda()
    counts={}
    for role in ['train','validation']:
        ns=[sum(len(np.load(x/'coordinates.npy',mmap_mode='r')) for x in paths_for(role,T)) for T in temperatures]
        total=sum(ns);counts[role]=total
        zarr=np.lib.format.open_memmap(out/f'{role}_z.npy',mode='w+',dtype=np.float32,shape=(total,24381))
        rarr=np.lib.format.open_memmap(out/f'{role}_residual.npy',mode='w+',dtype=np.float32,shape=(total,24381))
        tarr=np.lib.format.open_memmap(out/f'{role}_T.npy',mode='w+',dtype=np.float32,shape=(total,))
        offset=0
        for T in temperatures:
            for path in paths_for(role,T):
                x=np.load(path/'coordinates.npy',mmap_mode='r');f=np.load(path/'forces.npy',mmap_mode='r')
                for start in range(0,len(x),128):
                    end=min(start+128,len(x));batch=end-start
                    xx=torch.tensor(np.asarray(x[start:end]).reshape(batch,-1),device='cuda')
                    ff=torch.tensor(np.asarray(f[start:end]),device='cuda')
                    ff=ff+ff.sum(1,keepdim=True)/(params['full_atoms']-8127)
                    temp=torch.full((batch,),float(T),device='cuda')
                    z=wh.whiten(xx,temp)
                    score=wh.force_to_z(ff.reshape(batch,-1),temp)/(KB*temp[:,None])
                    zarr[offset:offset+batch]=z.cpu().numpy()
                    rarr[offset:offset+batch]=(-score-z).cpu().numpy()
                    tarr[offset:offset+batch]=T;offset+=batch
        for a in [zarr,rarr,tarr]:a.flush()
    write_json(out/'complete.json',dict(at=now(),counts=counts,preconditioner_sha256=sha256(out/'preconditioner.pt')))


class Preconditioner(nn.Module):
    def __init__(self):
        super().__init__()
        d=torch.load(ROOT/'data/processed/preconditioner.pt',map_location='cpu',weights_only=False)
        for name in ['P','sqrt_diagonal','expansion','contraction','means','temperatures','qref','sites']:
            self.register_buffer(name,d[name],persistent=False)
        self.dimension=len(d['qref'])
        self.scale0=float(torch.sqrt((d['sqrt_diagonal']**2).mean()))

    def mean(self,T):
        j=torch.searchsorted(self.temperatures,T).clamp(1,len(self.temperatures)-1)
        w=((T-self.temperatures[j-1])/(self.temperatures[j]-self.temperatures[j-1])).clamp(0,1)
        return self.means[j-1]*(1-w[:,None])+self.means[j]*w[:,None]

    def M(self,x,inverse=False):
        c=self.contraction if inverse else self.expansion
        with torch.autocast('cuda',enabled=False):
            x=x.float()
            return x+((x@self.P)*c)@self.P.T

    def physical(self,z,T):
        return self.mean(T)+self.M(z)*self.sqrt_diagonal*T.sqrt()[:,None]

    def whiten(self,x,T):
        return self.M((x-self.mean(T))/(T.sqrt()[:,None]*self.sqrt_diagonal),inverse=True)

    def force_to_z(self,f,T):
        return self.M(f*self.sqrt_diagonal)*T.sqrt()[:,None]


class Block(nn.Module):
    def __init__(self,c,embed=96):
        super().__init__();self.a=nn.Conv3d(c,c,3,padding=1);self.b=nn.Conv3d(c,c,3,padding=1)
        self.condition=nn.Linear(embed,2*c)
        nn.init.zeros_(self.condition.weight);nn.init.zeros_(self.condition.bias)
    def forward(self,x,c):
        scale,shift=self.condition(c).chunk(2,dim=-1)
        # Per-sample, nonparametric normalization controls activation scale.
        h=self.a(torch.nn.functional.silu(torch.nn.functional.group_norm(x,8,eps=1e-5)))
        h=torch.nn.functional.group_norm(h,8,eps=1e-5)
        h=h*(1+.1*scale[:,:,None,None,None])+.1*shift[:,:,None,None,None]
        return x+.25*self.b(torch.nn.functional.silu(h))


class Score3D(nn.Module):
    def __init__(self,width=24):
        super().__init__();self.wh=Preconditioner();self.config=dict(dimension=self.wh.dimension,width=width,
            architecture='bcc_unet3d_gn',normalization=dict(groups=8,epsilon=1e-5,affine=False))
        sites=self.wh.sites.numpy();cells=np.floor(sites).astype(int)+8
        basis=np.rint(2*(sites[:,0]-np.floor(sites[:,0]))).astype(int)
        index=(cells[:,0]*16+cells[:,1])*16+cells[:,2]
        self.register_buffer('basis',torch.tensor(basis),persistent=False)
        self.register_buffer('index',torch.tensor(index),persistent=False)
        for b in range(2):
            mask=basis==b
            self.register_buffer(f'atom_ids_{b}',torch.tensor(np.flatnonzero(mask)),persistent=False)
            self.register_buffer(f'cell_ids_{b}',torch.tensor(index[mask]),persistent=False)
        feature=torch.zeros(8,4096)
        for b in range(2):
            mask=basis==b
            feature[3*b:3*b+3,index[mask]]=self.wh.qref.reshape(-1,3)[mask].T/24
            feature[6+b,index[mask]]=1
        self.register_buffer('features',feature.reshape(8,16,16,16),persistent=False)
        self.register_buffer('frequency',torch.arange(1,17)*math.pi,persistent=False)
        self.embedding=nn.Sequential(nn.Linear(36,96),nn.SiLU(),nn.Linear(96,96))
        self.input=nn.Conv3d(14,width,3,padding=1)
        self.level0=nn.ModuleList([Block(width) for _ in range(2)])
        self.down0=nn.Conv3d(width,2*width,3,stride=2,padding=1)
        self.level1=nn.ModuleList([Block(2*width) for _ in range(2)])
        self.down1=nn.Conv3d(2*width,4*width,3,stride=2,padding=1)
        self.mid=nn.ModuleList([Block(4*width) for _ in range(2)])
        self.up1=nn.Conv3d(6*width,2*width,3,padding=1);self.up0=nn.Conv3d(3*width,width,3,padding=1)
        self.final=Block(width);self.output=nn.Conv3d(width,6,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,z,t,T):
        physical=self.wh.physical(z.float(),T)
        u=(physical-self.wh.qref)/(T.sqrt()[:,None]*self.wh.scale0)
        grid=torch.zeros(len(z),2,3,4096,device=z.device,dtype=z.dtype)
        for b in range(2):
            atom_ids=getattr(self,f'atom_ids_{b}');cell_ids=getattr(self,f'cell_ids_{b}')
            grid[:,b,:,cell_ids]=u.reshape(len(z),-1,3)[:,atom_ids].transpose(1,2)
        inputs=torch.cat((grid.reshape(len(z),6,16,16,16),self.features[None].expand(len(z),-1,-1,-1,-1)),dim=1)
        phase=t[:,None]*self.frequency;ratio=T/300
        c=self.embedding(torch.cat((phase.sin(),phase.cos(),t[:,None],ratio.log()[:,None],
                                    (ratio-1)[:,None],ratio.sqrt()[:,None]),dim=1))
        x=self.input(inputs)
        for block in self.level0:x=block(x,c)
        y=self.down0(x)
        for block in self.level1:y=block(y,c)
        b=self.down1(y)
        for block in self.mid:b=block(b,c)
        y=self.up1(torch.cat((y,torch.nn.functional.interpolate(b,size=y.shape[-3:],mode='trilinear',align_corners=False)),dim=1))
        x=self.up0(torch.cat((x,torch.nn.functional.interpolate(y,size=x.shape[-3:],mode='trilinear',align_corners=False)),dim=1))
        final=torch.nn.functional.group_norm(self.final(x,c),8,eps=1e-5)
        h=self.output(torch.nn.functional.silu(final)).reshape(len(z),2,3,4096)
        rows=torch.empty(len(z),8127,3,device=z.device,dtype=h.dtype)
        for b in range(2):
            atom_ids=getattr(self,f'atom_ids_{b}');cell_ids=getattr(self,f'cell_ids_{b}')
            rows[:,atom_ids]=h[:,b,:,cell_ids].transpose(1,2)
        # Chain rule back to z coordinates; temperature factors cancel scale normalization.
        return self.wh.M(rows.float().reshape(len(z),-1)*self.wh.sqrt_diagonal)/self.wh.scale0


def losses(pred,z,label,eps,alpha,sigma):
    return (pred-label).square().mean(),(sigma*z+alpha*sigma*pred-eps).square().mean()


def train(objective):
    p=protocol();folder=ROOT/'models'/objective;folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():return
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(8921)
    model=Score3D().cuda();ema=copy.deepcopy(model).eval()
    initial_hash=tensor_hash(model.state_dict().items())
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5)
    out=ROOT/'data/processed'
    z=torch.from_numpy(np.load(out/'train_z.npy')).cuda();r=torch.from_numpy(np.load(out/'train_residual.npy')).cuda()
    T=torch.from_numpy(np.load(out/'train_T.npy')).cuda()
    vz=torch.from_numpy(np.load(out/'validation_z.npy')).cuda();vr=torch.from_numpy(np.load(out/'validation_residual.npy')).cuda()
    vT=torch.from_numpy(np.load(out/'validation_T.npy')).cuda()
    generator=torch.Generator(device='cuda').manual_seed(731337)
    start_step=0;elapsed=0;history=[]
    # Common, fixed calibration uses only training data at identical initialization.
    cal_generator=torch.Generator(device='cuda').manual_seed(7123)
    scales=[]
    with torch.no_grad():
        for _ in range(8):
            ix=torch.randint(len(z),(64,),device='cuda',generator=cal_generator)
            tau=torch.rand(64,device='cuda',generator=cal_generator)
            a=torch.cos(math.pi*tau/2)[:,None];s=torch.sin(math.pi*tau/2)[:,None]
            eps=torch.randn(z[ix].shape,device='cuda',generator=cal_generator);noisy=a*z[ix]+s*eps
            scales.append([float(x) for x in losses(model(noisy,tau,T[ix]),noisy,r[ix],eps,a,s)])
    scale=np.mean(scales,axis=0)
    identity=dict(objective=objective,initial_state_sha256=initial_hash,
        preconditioner_sha256=sha256(out/'preconditioner.pt'),
        training_setup=p['models'],training_frames=len(z),validation_frames=len(vz),
        initialization_seed=8921,training_draw_seed=731337,loss_scales=scale.tolist(),
        learning_source_sha256=LEARNING_SOURCE_SHA256,model_config=model.config)
    warm_state=None
    if objective=='force_then_denoising':
        decision=json.loads((ROOT/'results/switch_selection.json').read_text())
        if decision['validation_temperatures']!=p['temperatures_K']['train']:
            raise RuntimeError('Switch selection temperatures do not match the frozen protocol.')
        parent_path=ROOT/decision['parent_resume_checkpoint']
        if sha256(parent_path)!=decision['parent_resume_sha256']:
            raise RuntimeError('Force warm-start checkpoint changed after selection.')
        warm_state=torch.load(parent_path,map_location='cuda',weights_only=False)
        parent_identity=json.loads((ROOT/'models/force/initialization.json').read_text())
        if warm_state['step']!=decision['selected_step'] or warm_state['objective']!='force':
            raise RuntimeError('Wrong parent checkpoint for the switching branch.')
        identity['first_training_draw_sha256']=parent_identity['first_training_draw_sha256']
        identity['warm_start']=dict(parent='force',step=warm_state['step'],
            parent_resume_checkpoint=decision['parent_resume_checkpoint'],
            parent_resume_sha256=decision['parent_resume_sha256'],
            cumulative_parent_seconds=warm_state['elapsed_s'],
            post_switch_objective='unscaled denoising MSE')
        for checkpoint_step in p['models']['checkpoint_steps']:
            if checkpoint_step>warm_state['step']:continue
            source=ROOT/'models/force'/f'step_{checkpoint_step:06d}.pt'
            target=folder/source.name
            if not target.exists():target.symlink_to(source)
    manifest=folder/'initialization.json'
    if manifest.exists():
        prior=json.loads(manifest.read_text())
        if any(prior.get(key)!=value for key,value in identity.items()):
            raise RuntimeError('Training initialization or setup changed; review before resuming.')
        identity=prior
    for other in p['models']['objectives']:
        other_path=ROOT/'models'/other/'initialization.json'
        if other!=objective and other_path.exists():
            prior=json.loads(other_path.read_text())
            for key in ['initial_state_sha256','preconditioner_sha256','training_setup',
                        'training_frames','validation_frames','loss_scales','learning_source_sha256','model_config']:
                if prior[key]!=identity[key]:
                    raise RuntimeError(f'Matched-model setup differs for {key}: {other} versus {objective}')
    write_json(manifest,identity)
    if (folder/'last.pt').exists() or warm_state is not None:
        state=torch.load(folder/'last.pt',map_location='cuda',weights_only=False) if (folder/'last.pt').exists() else warm_state
        if state['preconditioner_sha256']!=sha256(out/'preconditioner.pt'):raise RuntimeError('Preconditioner changed at resume.')
        if state['initial_state_sha256']!=initial_hash:raise RuntimeError('Initial model changed at resume.')
        if state['config']!=model.config or state['learning_source_sha256']!=LEARNING_SOURCE_SHA256:
            raise RuntimeError('Network implementation changed at resume.')
        model.load_state_dict(state['model']);ema.load_state_dict(state['ema']);optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['rng'].cpu());start_step=state['step'];elapsed=state['elapsed_s'];history=state['history']
        scale=state['loss_scales']
        if objective=='force_then_denoising' and not (folder/'last.pt').exists():
            inherited=dict(model=tensor_hash(model.state_dict().items()),ema=tensor_hash(ema.state_dict().items()),
                rng=tensor_hash([('rng',generator.get_state())]))
            expected=dict(model=tensor_hash(state['model'].items()),ema=tensor_hash(state['ema'].items()),
                rng=tensor_hash([('rng',state['rng'])]))
            if inherited!=expected:raise RuntimeError('Warm-start model, EMA, or RNG was not inherited exactly.')
            for key,values in optimizer.state_dict()['state'].items():
                for name,value in values.items():
                    other=state['optimizer']['state'][key][name]
                    if torch.is_tensor(value):
                        if not torch.equal(value.cpu(),other.cpu()):raise RuntimeError('Warm-start optimizer state mismatch.')
                    elif value!=other:raise RuntimeError('Warm-start optimizer state mismatch.')
            write_json(folder/'warm_start_validation.json',dict(passed=True,at=now(),
                step=start_step,elapsed_s=elapsed,inherited_hashes=inherited,optimizer_exact=True,
                source=identity['warm_start'],post_switch_objective='denoising',learning_rate_schedule_reset=False))
    del warm_state
    maximum=p['models']['updates_per_model_max'];wall=time.monotonic()
    for step in range(start_step+1,maximum+1):
        ix=torch.randint(len(z),(64,),device='cuda',generator=generator)
        tau=torch.rand(64,device='cuda',generator=generator)
        eps=torch.randn((64,model.config['dimension']),device='cuda',generator=generator)
        if step==1:
            draw_hash=tensor_hash([('indices',ix),('time',tau),('noise',eps)])
            for other in p['models']['objectives']:
                other_path=ROOT/'models'/other/'initialization.json'
                if other!=objective and other_path.exists():
                    prior=json.loads(other_path.read_text())
                    if prior.get('first_training_draw_sha256')!=draw_hash:
                        raise RuntimeError('First training draw differs across the three models.')
            identity['first_training_draw_sha256']=draw_hash;write_json(manifest,identity)
        a=torch.cos(math.pi*tau/2)[:,None];s=torch.sin(math.pi*tau/2)[:,None];noisy=a*z[ix]+s*eps
        optimizer.zero_grad(set_to_none=True)
        f,n=losses(model(noisy,tau,T[ix]),noisy,r[ix],eps,a,s)
        loss={'force':f,'denoising':n,'force_then_denoising':n}[objective]
        if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite training loss; no checkpoint selection allowed.')
        loss.backward();gradient_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True)
        lr=2e-4*(.1+.9*(1+math.cos(math.pi*step/maximum))/2)
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.step()
        with torch.no_grad():
            for average,current in zip(ema.parameters(),model.parameters()):average.lerp_(current,.001)
        if step%1000==0 or step in p['models']['checkpoint_steps']:
            with torch.no_grad():
                val=[];baseline=[];residual_squares=[];activation_zeros=[]
                def activation_audit(module,inputs):
                    activation_zeros.append(float((inputs[0]==0).float().mean()))
                handle=ema.output.register_forward_pre_hook(activation_audit)
                vg=torch.Generator(device='cuda').manual_seed(4413)
                for _ in range(8):
                    j=torch.randint(len(vz),(64,),device='cuda',generator=vg);t=torch.rand(64,device='cuda',generator=vg)
                    av=torch.cos(math.pi*t/2)[:,None];sv=torch.sin(math.pi*t/2)[:,None]
                    ev=torch.randn((64,model.config['dimension']),device='cuda',generator=vg);zz=av*vz[j]+sv*ev
                    prediction=ema(zz,t,vT[j])
                    val.append([float(x) for x in losses(prediction,zz,vr[j],ev,av,sv)])
                    residual_squares.append(float(prediction.square().mean()))
                    baseline.append([float(x) for x in losses(torch.zeros_like(prediction),zz,vr[j],ev,av,sv)])
                handle.remove()
            row=dict(step=step,elapsed_s=elapsed+time.monotonic()-wall,train_force_loss=float(f),train_denoising_loss=float(n),
                     validation_force_loss=float(np.mean(val,axis=0)[0]),validation_denoising_loss=float(np.mean(val,axis=0)[1]),
                     validation_residual_rms=float(np.sqrt(np.mean(residual_squares))),
                     validation_output_input_zero_fraction=float(np.mean(activation_zeros)),
                     validation_zero_residual_force_loss=float(np.mean(baseline,axis=0)[0]),
                     validation_zero_residual_denoising_loss=float(np.mean(baseline,axis=0)[1]),
                     gradient_norm_before_clip=float(gradient_norm),
                     training_phase='denoising' if objective=='force_then_denoising' else objective)
            history.append(row)
            state=dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),rng=generator.get_state(),
                step=step,elapsed_s=row['elapsed_s'],history=history,objective=objective,config=model.config,loss_scales=scale,
                preconditioner_sha256=sha256(out/'preconditioner.pt'),learning_source_sha256=LEARNING_SOURCE_SHA256)
            state['initial_state_sha256']=initial_hash
            if 'warm_start' in identity:state['warm_start']=identity['warm_start']
            torch.save(state,folder/'last.pending.pt');(folder/'last.pending.pt').replace(folder/'last.pt')
            if step in p['models']['checkpoint_steps']:
                torch.save(dict(ema=ema.state_dict(),step=step,elapsed_s=row['elapsed_s'],config=model.config,
                    objective=objective,preconditioner_sha256=state['preconditioner_sha256'],
                    learning_source_sha256=LEARNING_SOURCE_SHA256),folder/f'step_{step:06d}.pt')
                if objective=='force' and step<=p['models']['warm_start']['early_window_max_step']:
                    torch.save(state,folder/f'resume_step_{step:06d}.pt')
            write_json(folder/'history.json',history);print(objective,json.dumps(row),flush=True)
    write_json(folder/'complete.json',dict(at=now(),objective=objective,steps=maximum,loss_scales=list(map(float,scale)),
        initial_state_sha256=initial_hash,first_training_draw_sha256=identity['first_training_draw_sha256'],
        final_training_rng_sha256=tensor_hash([('training_rng',generator.get_state())])))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preprocess','force','denoising','force_then_denoising'])
    args=parser.parse_args()
    preprocess() if args.stage=='preprocess' else train(args.stage)
