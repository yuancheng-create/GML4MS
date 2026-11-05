"""Resumable independent MD trajectories and autocorrelation-aware summaries."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
from common import ROOT, protocol, write_json, now, sha256
from physics import EAM,bcc,stress,buffer_margin


def geometry(n,shear,threads=8):
    p=protocol();a=p['potential']['a0_verified_A']
    F=np.array(p['geometry']['F']);F[0,1]=shear
    path=ROOT/'checks'/f'n{n}_shear{shear:.3f}_relaxed.npz'
    if not path.exists():
        from backend import Simulation
        q,sites=bcc(n,a,F)
        sim=Simulation(q,F,n*a,threads=threads,logfile=path.with_suffix('.log'))
        relaxed=sim.relax()
        maximum=float(np.linalg.norm(sim.forces(),axis=1).max())
        if maximum>1e-3:
            sim.close();raise RuntimeError('Insufficiently relaxed initial condition.')
        np.savez(path,q=relaxed,qref=q,sites=sites,F=F,a=a)
        write_json(path.with_suffix('.json'),dict(n=n,shear=shear,max_force_eV_per_A=maximum,completed_at=now()))
        sim.close()
    return dict(np.load(path))


def trajectory(n,T,role,seed,production_ps=200,equilibration_ps=50,dt=.001,
               root=None):
    p=protocol();shear=p['geometry']['F'][0][1]
    if not json.loads((ROOT/'checks/physics_validation.json').read_text()).get('passed'):
        raise RuntimeError('Run src/prepare.py before MD; physics validation is required.')
    folder=Path(root or ROOT/'data')/f'{role}_n{n}_T{T}_s{seed}_dt{dt:.4f}'
    if folder.is_symlink():
        raise RuntimeError('Generate local trajectories; external data symlinks are not accepted.')
    folder.mkdir(parents=True,exist_ok=True)
    meta=folder/'meta.json'
    previous=json.loads(meta.read_text()) if meta.exists() else {}
    if previous.get('failure'):
        raise RuntimeError(f'Previous physical failure requires review: {folder}')
    if previous.get('complete') and previous['production_ps']>=production_ps:
        return folder
    production_ps=max(production_ps,previous.get('production_ps',0))
    extensions=previous.get('extensions',[])
    if previous and production_ps>previous['production_ps']:
        extensions=extensions+[dict(at=now(),from_ps=previous['production_ps'],to_ps=production_ps)]
    if previous:
        write_json(meta,dict(previous,complete=False,production_ps=production_ps,extensions=extensions,
                             status='preparing_resume'))
    geo=geometry(n,shear);q0=geo['q'];F=geo['F'];a=float(geo['a']);sites=geo['sites']
    ids=np.flatnonzero(np.all((sites>=-8)&(sites<8),axis=1)) if n>16 else np.arange(len(q0))
    frames=round(production_ps/.1);stride=round(.1/dt);equil=round(equilibration_ps/dt)
    e=EAM();large=n>6
    far=np.linalg.norm(sites*a,axis=1)>2*a+2*e.cutoff
    if not np.any(far):far=np.linalg.norm(sites*a,axis=1)>2*a+e.cutoff/2
    resume=folder/'restart.npz'
    arrays={}
    for name,shape in [('coordinates',(frames,len(ids),3)),('forces',(frames,len(ids),3)),('statistics',(frames,16))]:
        path=folder/f'{name}.npy'
        if path.exists():
            arrays[name]=np.load(path,mmap_mode='r+')
            if arrays[name].shape!=shape:
                old=arrays[name]
                if old.shape[1:]!=shape[1:] or old.shape[0]>shape[0]:
                    raise RuntimeError('Cannot shrink or change the geometry of a started MD job.')
                # Extend atomically; retain every previously saved frame and the RNG state.
                pending=path.with_suffix('.extend.npy')
                expanded=np.lib.format.open_memmap(pending,mode='w+',dtype=old.dtype,shape=shape)
                for begin in range(0,len(old),64):
                    end=min(begin+64,len(old));expanded[begin:end]=old[begin:end]
                expanded.flush();del expanded,old;del arrays[name]
                pending.replace(path);arrays[name]=np.load(path,mmap_mode='r+')
        else:
            arrays[name]=np.lib.format.open_memmap(path,mode='w+',dtype=np.float64 if name=='statistics' else np.float32,shape=shape)
    first=0;done_equil=0;previous_seconds=0
    audit=json.loads((ROOT/'checks/gpu_backend_validation.json').read_text())
    if not audit['passed']: raise RuntimeError('GPU backend not validated.')
    from gpu_md import GPUSimulation
    sim=GPUSimulation(q0,F,n*a,T,seed,dt=dt)
    if resume.exists():
        state=np.load(resume)
        sim.q[:]=sim.cp.asarray(state['q']);sim.v[:]=sim.cp.asarray(state['v']);sim.step[:]=sim.cp.asarray(state['step'])
        first=int(state['frame']);done_equil=int(state['equil']);previous_seconds=float(state['seconds'])
        sim.rebuild();sim.evaluate()
    start=time.monotonic()
    info=dict(role=role,n=n,temperature_K=T,seed=seed,dt_ps=dt,backend='gpu',
              production_ps=production_ps,equilibration_ps=equilibration_ps,save_interval_ps=.1,
              reference_atoms=len(q0),retained_atoms=len(ids),frames=frames,frames_written=first,
              potential_sha256=sha256(ROOT/'potentials/Fe_Mishin2006.eam.alloy'),complete=False,
              statistics_columns=['xx','xy','xz','yx','yy','yz','zx','zy','zz','temperature_K',
                  'energy_eV_per_atom','core_atoms','min_pair_distance_A','buffer_slack_A','far_rms_displacement_A','time_ps'])
    info['maximum_far_fraction_displacement_gt_1A']=previous.get('maximum_far_fraction_displacement_gt_1A',0)
    info['backend_sources']=sim.source_hashes
    info['extensions']=extensions
    write_json(meta,info)

    def save_state(frame,equilibrated):
        for arr in arrays.values():arr.flush()
        elapsed=previous_seconds+time.monotonic()-start
        tmp=folder/'restart.pending.npz'
        np.savez(tmp,q=sim.q.get(),v=sim.v.get(),step=sim.step.get(),frame=frame,equil=equilibrated,seconds=elapsed)
        tmp.replace(resume)
        info.update(frames_written=frame,equilibration_steps_completed=equilibrated,elapsed_s=elapsed,updated_at=now())
        write_json(meta,info)

    while done_equil<equil:
        count=min(round(5/dt),equil-done_equil)
        sim.run(count);done_equil+=count;save_state(first,done_equil)
        print(folder.name,'equilibrated_ps',round(done_equil*dt,3),flush=True)
    for frame in range(first,frames):
        sim.run(stride)
        q=sim.positions();local=q[ids]
        v=e.evaluate(local) if large else e.evaluate(local,F,n*a)
        sigma,count,_=stress(local,v,F,a,T,whole_periodic=not large)
        obs=sim.observables()
        margin=buffer_margin(q,ids,F,a,e.cutoff) if large else np.nan
        mindist=float(v['distances'].min())
        displacement=np.linalg.norm(q-q0,axis=1)
        far_rms=float(np.sqrt(np.mean(displacement[far]**2)))
        far_fraction=float(np.mean(displacement[far]>1))
        info['maximum_far_fraction_displacement_gt_1A']=max(info.get('maximum_far_fraction_displacement_gt_1A',0),far_fraction)
        arrays['coordinates'][frame]=local
        arrays['forces'][frame]=sim.forces()[ids]
        arrays['statistics'][frame]=np.r_[sigma.ravel(),obs['temperature_K'],obs['energy_eV_per_atom'],
                  count,mindist,margin,far_rms,(frame+1)*.1]
        if mindist<1.5 or (large and margin<=0) or far_rms>=.75 or far_fraction>=.005 or not np.isfinite(sigma).all():
            save_state(frame+1,done_equil)
            info['failure']='physical-domain or buffer check failed; sample retained and run halted'
            write_json(meta,info)
            raise RuntimeError(info['failure'])
        if (frame+1)%100==0 or frame+1==frames:
            save_state(frame+1,done_equil)
            print(folder.name,'frames',frame+1,'of',frames,'elapsed',round(info['elapsed_s'],1),flush=True)
    info['complete']=True;info['completed_at']=now();write_json(meta,info)
    np.savez(folder/'geometry.npz',qref=geo['qref'][ids],sites=sites[ids],F=F,a=a,full_atoms=len(q0))
    return folder


def summarize(paths):
    if not paths:raise RuntimeError('No completed MD chains available.')
    series=[np.load(Path(p)/'statistics.npy',mmap_mode='r')[:,:9].reshape(-1,3,3) for p in paths]
    means=np.array([x.mean(axis=0) for x in series])
    if len(set(map(len,series)))!=1:raise RuntimeError('Chain lengths must match for equal-chain averaging.')
    audits=[];covariances=[];halfwidths=[]
    for block_ps in [5,10,20,40]:
        length=round(block_ps/.1);blocks=[]
        if min(map(len,series))<4*length:continue
        for x in series:
            n=len(x)//length
            blocks.extend(x[:n*length].reshape(n,length,3,3).mean(axis=1))
        b=np.asarray(blocks)
        covariance=np.cov(b.reshape(len(b),9),rowvar=False)/len(b)
        half=student_t.ppf(.975,len(b)-1)*np.sqrt(np.diag(covariance)).reshape(3,3)
        audits.append(dict(block_ps=block_ps,blocks=len(b),ci95_halfwidth=half.tolist()))
        covariances.append(covariance);halfwidths.append(half)
    if len(means)>1:
        hchain=student_t.ppf(.975,len(means)-1)*means.std(axis=0,ddof=1)/np.sqrt(len(means))
        halfwidths.append(hchain)
        covariances.append(np.cov(means.reshape(len(means),9),rowvar=False)/len(means))
    half=np.max(halfwidths,axis=0)
    covariance=max(covariances,key=np.trace).copy()
    # PSD diagonal inflation retains conservative componentwise variance in selection.
    target=np.max([np.diag(c) for c in covariances],axis=0)
    covariance+=np.diag(np.maximum(target-np.diag(covariance),0))
    # Initial-positive-sequence estimate for stress autocorrelation, reported separately.
    taus=[]
    for x in series:
        values=x.reshape(len(x),9);values=values-values.mean(0)
        ft=np.fft.rfft(values,n=2*len(x),axis=0)
        ac=np.fft.irfft(ft*ft.conj(),n=2*len(x),axis=0)[:len(x)].real
        ac/=np.arange(len(x),0,-1)[:,None];ac/=np.maximum(ac[0],1e-30)
        tau=np.ones(9)
        for j in range(9):
            for lag in range(1,min(len(x)-1,len(x)//2),2):
                pair=ac[lag,j]+ac[lag+1,j]
                if pair<=0:break
                tau[j]+=2*pair
        taus.append(tau)
    return dict(mean=means.mean(axis=0).tolist(),ci95_halfwidth=half.tolist(),chain_means=means.tolist(),
                frames=sum(map(len,series)),chains=len(series),blocking_audit=audits,
                integrated_autocorrelation_ps=(.1*np.array(taus)).tolist(),
                approximate_effective_samples=(sum(map(len,series))/np.max(taus,axis=0)).reshape(3,3).tolist(),
                covariance_of_mean=covariance.tolist(),
                uncertainty='maximum of 5/10/20/40-ps batch-mean and independent-chain t intervals')
