"""Validation-only checkpoint/sampler selection, then independent final evaluation."""
import json
import time
import math
from pathlib import Path
import numpy as np
import torch
from common import ROOT,protocol,write_json,now,sha256
from observable_parallel import evaluate_observables
from learning import Score3D,paths_for,concatenate,LEARNING_SOURCE_SHA256
from diffusion import sample_latent
from md_jobs import summarize

torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False


def load_model(objective,step):
    model=Score3D().cuda().eval()
    state=torch.load(ROOT/'models'/objective/f'step_{step:06d}.pt',map_location='cuda',weights_only=False)
    if state['preconditioner_sha256']!=sha256(ROOT/'data/processed/preconditioner.pt'):
        raise RuntimeError('Checkpoint and preconditioner differ; cached samples cannot be reused.')
    if state.get('config')!=model.config or state.get('learning_source_sha256')!=LEARNING_SOURCE_SHA256:
        raise RuntimeError('Checkpoint and network implementation differ; use the matching training source.')
    model.load_state_dict(state['ema'])
    return model,state


def gaussian_samples(T,count,seed,start_at=0):
    temps=protocol()['temperatures_K']['train'];j=np.clip(np.searchsorted(temps,T),1,len(temps)-1)
    lo,hi=temps[j-1],temps[j];w=float(np.clip((T-lo)/(hi-lo),0,1))
    weights=[(lo,1-w),(hi,w)]
    rng=torch.Generator(device='cuda').manual_seed(seed)
    factors=[];mean=None;trace=0
    for temp,weight in weights:
        if weight<=0:continue
        x=concatenate(paths_for('train',temp),'coordinates').reshape(-1,24381)
        mu=x.mean(0,dtype=np.float64).astype(np.float32);mean=weight*mu if mean is None else mean+weight*mu
        dx=torch.tensor(x-mu,device='cuda')
        factor=dx*math.sqrt(weight*T/temp/len(dx))
        factors.append(factor);trace+=float(factor.square().sum()/24381)
    # Full empirical covariance factor; no PCA truncation in this comparison.
    floor=trace*1e-6
    for start in range(0,count,64):
        size=min(64,count-start)
        x=torch.tensor(mean,device='cuda')[None].expand(size,-1).clone()
        for factor in factors:
            x.add_(torch.randn(size,len(factor),device='cuda',generator=rng)@factor)
        x.add_(math.sqrt(floor)*torch.randn(size,24381,device='cuda',generator=rng))
        # Replaying the cheap random draws preserves the exact stream on resume;
        # previously committed configurations do not have their stress recomputed.
        if start>=start_at:yield x.cpu().numpy().reshape(size,8127,3)


def generated(objective,step,T,count,steps=100,amp=False,purpose='validation',seed=811872):
    if objective=='force_then_denoising':
        decision=json.loads((ROOT/'results/switch_selection.json').read_text())
        if step<=decision['selected_step']:
            result=generated('force',step,T,count,steps,amp,purpose,seed)
            return dict(result,method=objective,shared_prefix_source='force')
    label=f'{objective}_s{step:06d}_T{T}_n{count}_heun{steps}_'+('bf16' if amp else 'fp32')+f'_seed{seed}'
    folder=ROOT/'results'/purpose/label;folder.mkdir(parents=True,exist_ok=True)
    p=protocol()
    if purpose in ['validation','checkpoint_validation','sampler_validation'] and T not in p['temperatures_K']['train']:
        raise ValueError('Held-out temperatures cannot enter checkpoint or sampler selection.')
    identity=dict(geometry=p['geometry'],potential_sha256=p['potential']['sha256'],
        preconditioner_sha256=sha256(ROOT/'data/processed/preconditioner.pt'),
        physics_sha256=sha256(Path(__file__).with_name('physics.py')),
        observable_backend_sha256=sha256(Path(__file__).with_name('observable_parallel.py')),
        observable_workers=p['generation'].get('observable_workers',4),
        sampler_sha256=sha256(Path(__file__).with_name('diffusion.py')),
        learning_source_sha256=LEARNING_SOURCE_SHA256 if objective!='gaussian' else None,
        model_sha256=sha256(ROOT/'models'/objective/f'step_{step:06d}.pt') if objective!='gaussian' else None)
    manifest=folder/'input_identity.json'
    if manifest.exists() and json.loads(manifest.read_text())!=identity:
        raise RuntimeError('Inputs changed since sampling started; preserve old outputs and review before resuming.')
    write_json(manifest,identity)
    if (folder/'summary.json').exists():return json.loads((folder/'summary.json').read_text())
    if (folder/'failure.json').exists():
        raise RuntimeError(f'Previous sampling failure requires review: {folder}/failure.json')
    progress=json.loads((folder/'progress.json').read_text()) if (folder/'progress.json').exists() else {}
    first=progress.get('samples',0);previous_seconds=progress.get('elapsed_s',0.)
    save_coordinates=purpose=='final' and T==200
    coordinates_store=None
    if first and first!=count and first%(64 if objective=='gaussian' else 512):
        raise RuntimeError('Saved progress is not at a committed sampling batch boundary.')
    if first:
        stat=np.load(folder/'stress.npy',mmap_mode='r+')
        if stat.shape!=(count,3,3):raise RuntimeError('Saved stress array has an incompatible shape.')
        snapshots=list(np.load(folder/'snapshots.npz')['q'])
        if save_coordinates:coordinates_store=np.load(folder/'coordinates.npy',mmap_mode='r+')
    else:
        stat=np.lib.format.open_memmap(folder/'stress.npy',mode='w+',dtype=np.float64,shape=(count,3,3));snapshots=[]
    F=np.array(p['geometry']['F']);a=p['potential']['a0_verified_A']
    model=None
    if objective!='gaussian':model,_=load_model(objective,step)
    def neural_batches():
        for start in range(first,count,512):
            size=min(512,count-start)
            z=sample_latent(model,T,size,steps=steps,batch=64,seed=seed+start,amp=amp,cuda_graph=True)
            with torch.no_grad():
                temp=torch.full((size,),float(T),device='cuda')
                yield model.wh.physical(torch.tensor(z,device='cuda'),temp).cpu().numpy().reshape(size,8127,3)
    start_time=time.monotonic();index=first;minimum=progress.get('min_pair_distance_A',float('inf'))
    q=None
    try:
        iterator=gaussian_samples(T,count,seed,start_at=first) if objective=='gaussian' else neural_batches()
        for batch in iterator:
            if save_coordinates and coordinates_store is None:
                coordinates_store=np.lib.format.open_memmap(folder/'coordinates.npy',mode='w+',
                    dtype=batch.dtype,shape=(count,*batch.shape[1:]))
            observables=iter(evaluate_observables(batch,F,a,T,workers=identity['observable_workers']))
            for coordinates in batch:
                q=coordinates.astype(np.float64)
                sigma,min_pair_distance=next(observables)
                stat[index]=sigma
                if coordinates_store is not None:coordinates_store[index]=coordinates
                minimum=min(minimum,min_pair_distance)
                if not np.isfinite(sigma).all():raise RuntimeError('Nonfinite generated stress; no samples filtered.')
                if index<4:snapshots.append(q)
                index+=1
            stat.flush()
            if coordinates_store is not None:coordinates_store.flush()
            if not (folder/'snapshots.npz').exists():
                pending=folder/'snapshots.pending.npz'
                np.savez(pending,q=np.array(snapshots),F=F,a=a);pending.replace(folder/'snapshots.npz')
            write_json(folder/'progress.json',dict(samples=index,target=count,min_pair_distance_A=minimum,
                elapsed_s=previous_seconds+time.monotonic()-start_time,updated_at=now()))
            q=None
    except Exception as exc:
        stat.flush()
        failure=repr(exc);write_json(folder/'failure.json',dict(at=now(),failure=failure,samples_written=index,
            failed_sample_index=index if q is not None else None,
            failure_stage='observable' if q is not None else 'sampling_before_coordinates_available',no_filtering=True))
        if q is not None:np.savez(folder/'failure_configuration.npz',q=q,index=index,F=F,a=a)
        raise
    mean=stat.mean(axis=0);cov=np.cov(np.asarray(stat).reshape(count,9),rowvar=False)/count
    result=dict(method=objective,checkpoint=step,temperature_K=T,samples=count,heun_steps=steps,
        network_precision='BF16' if amp else 'FP32',seed=seed,mean=mean.tolist(),
        ci95_halfwidth=(1.96*np.sqrt(np.diag(cov)).reshape(3,3)).tolist(),covariance_of_mean=cov.tolist(),
        min_pair_distance_A=minimum,elapsed_s=previous_seconds+time.monotonic()-start_time,folder=str(folder.relative_to(ROOT)),complete=True)
    if save_coordinates:result['coordinates_file']='coordinates.npy'
    np.savez(folder/'snapshots.npz',q=np.array(snapshots),F=F,a=a)
    write_json(folder/'summary.json',result)
    return result


def selection_score(reports,references):
    if not reports or len(reports)!=len(references):
        raise ValueError('Validation reports and references must match.')
    counts={x['samples'] for x in reports};seeds={x['seed'] for x in reports}
    if len(counts)!=1 or len(seeds)!=1 or next(iter(counts))<2:
        raise ValueError('Cross-temperature score uncertainty requires aligned sample counts and latent seeds.')
    count=next(iter(counts))
    deltas=[np.array(x['mean']).ravel()-np.array(y['mean']).ravel() for x,y in zip(reports,references)]
    weights=np.array([.5,.5,0.,0.,0.,0.,0.,0.,0.])
    score=float(np.sqrt(np.mean([np.dot(x*weights,x) for x in deltas])))
    if score==0:
        raise ValueError('Delta-method uncertainty is undefined at zero validation norm; review rather than report zero uncertainty.')
    variance=0
    if score>0:
        generated_projection=np.zeros(count,dtype=np.float64)
        for delta,x,y in zip(deltas,reports,references):
            gradient=weights*delta/(len(deltas)*score)
            values=np.load(ROOT/x['folder']/'stress.npy',mmap_mode='r')
            if values.shape!=(count,3,3):raise RuntimeError('Validation stress arrays are not aligned.')
            values=np.asarray(values).reshape(count,9)
            mean=np.array(x['mean']).ravel()
            if not np.allclose(values.mean(0),mean,rtol=1e-12,atol=1e-10):
                raise RuntimeError('Validation stress summary differs from its stored samples.')
            # Shared latent draws correlate temperatures. Project each aligned
            # sample first, then estimate its variance, retaining cross terms.
            generated_projection+=(values-mean)@gradient
            # Reference MD uses independent chains and seeds across temperatures.
            variance+=gradient@np.array(y['covariance_of_mean'])@gradient
        variance+=np.var(generated_projection,ddof=1)/count
    return score,float(np.sqrt(max(variance,0)))


def evaluate_checkpoint(objective,step,count,refs):
    reports=[generated(objective,step,T,count,purpose='checkpoint_validation') for T in refs]
    score,se=selection_score(reports,list(refs.values()))
    state=torch.load(ROOT/'models'/objective/f'step_{step:06d}.pt',map_location='cpu',weights_only=False)
    return dict(step=step,samples_per_T=count,score=score,score_standard_error=se,
        training_seconds=state['elapsed_s'],per_temperature=reports)


def select_checkpoints(objectives=None):
    p=protocol();out=ROOT/'results/checkpoint_selection';out.mkdir(parents=True,exist_ok=True)
    refs={T:summarize(paths_for('validation',T)) for T in p['temperatures_K']['train']}
    selection_path=out/'selection.json'
    chosen=json.loads(selection_path.read_text()) if selection_path.exists() else {}
    for objective in objectives or p['models']['all_objectives']:
        rows=[]
        def evaluate(step,count):
            return evaluate_checkpoint(objective,step,count,refs)
        for step in p['models']['coarse_stress_validation_steps']:
            rows.append(evaluate(step,2048));write_json(out/f'{objective}_progress.json',rows)
        if objective=='force_then_denoising':
            switch=json.loads((ROOT/'results/switch_selection.json').read_text())['selected_step']
            if not any(r['step']==switch for r in rows):rows.append(evaluate(switch,2048))
        best=min(rows,key=lambda x:x['score'])
        saved=p['models']['checkpoint_steps'];j=saved.index(best['step'])
        neighbors=saved[max(0,j-1):min(len(saved),j+2)]
        for step in neighbors:
            if not any(x['step']==step for x in rows):rows.append(evaluate(step,2048))
        contenders=sorted(rows,key=lambda x:x['score'])[:3]
        refined=[evaluate(x['step'],10000) for x in contenders]
        best=min(refined,key=lambda x:x['score'])
        # Fixed one-standard-error rule avoids selecting a later indistinguishable point.
        selected=min((x for x in refined if x['score']<=best['score']+best['score_standard_error']),key=lambda x:x['step'])
        chosen[objective]=dict(selected_step=selected['step'],coarse=sorted(rows,key=lambda x:x['step']),refined=refined,
            rule='earliest among top-three refined candidates within one standard error of best validation score',validation_temperatures=list(refs),
            score_uncertainty='delta method retaining cross-temperature generated covariance from aligned common-latent samples, plus independent validation-MD covariance')
        write_json(out/f'{objective}.json',chosen[objective]);print('selected',objective,selected['step'],flush=True)
    write_json(out/'selection.json',chosen)


def select_switch():
    p=protocol();window=p['models']['warm_start']['early_window_max_step']
    selected=json.loads((ROOT/'results/checkpoint_selection/selection.json').read_text())['force']
    refs={T:summarize(paths_for('validation',T)) for T in p['temperatures_K']['train']}
    rows=[r for r in selected['coarse'] if r['step']<=window]
    best=min(rows,key=lambda r:r['score'])
    saved=[s for s in p['models']['checkpoint_steps'] if s<=window]
    index=saved.index(best['step'])
    for step in saved[max(0,index-1):index+2]:
        if not any(r['step']==step for r in rows):rows.append(evaluate_checkpoint('force',step,2048,refs))
    refined=[evaluate_checkpoint('force',r['step'],10000,refs) for r in sorted(rows,key=lambda r:r['score'])[:3]]
    best=min(refined,key=lambda r:r['score'])
    chosen=min((r for r in refined if r['score']<=best['score']+best['score_standard_error']),key=lambda r:r['step'])
    parent=ROOT/'models/force'/f"resume_step_{chosen['step']:06d}.pt"
    result=dict(selected_step=chosen['step'],parent='force',validation_temperatures=list(refs),
        early_window_max_step=window,rule=p['models']['warm_start']['switch_selection'],
        coarse=sorted(rows,key=lambda r:r['step']),refined=refined,
        parent_resume_checkpoint=str(parent.relative_to(ROOT)),parent_resume_sha256=sha256(parent),
        held_out_temperature_used=False)
    write_json(ROOT/'results/switch_selection.json',result)
    print('Selected force-to-denoising switch:',chosen['step'],flush=True)


def select_sampler():
    p=protocol();selection=json.loads((ROOT/'results/checkpoint_selection/selection.json').read_text())
    path=ROOT/'results/sampler_selection.json'
    candidates=[]
    def compare(objective,T,step,reference_steps,test_steps,amp=False,count=2048):
        base=generated(objective,step,T,count,reference_steps,False,'sampler_validation')
        test=generated(objective,step,T,count,test_steps,amp,'sampler_validation')
        xb=np.load(ROOT/base['folder']/'stress.npy');xt=np.load(ROOT/test['folder']/'stress.npy')
        diff=xt-xb;mean=diff.mean(0);half=1.96*diff.std(0,ddof=1)/np.sqrt(len(diff))
        combined=np.sqrt(np.array(base['ci95_halfwidth'])**2+np.array(test['ci95_halfwidth'])**2)
        upper=abs(mean)+half
        ok=bool(np.all(upper<=.01) and np.all(abs(mean)<=combined+1e-12))
        # Increase paired samples only if the 0.01 GPa criterion is unresolved.
        if not ok and count==2048 and np.all(np.maximum(abs(mean)-half,0)<=.01):
            return compare(objective,T,step,reference_steps,test_steps,amp,count=10000)
        return dict(method=objective,T=T,mean_drift=mean.tolist(),paired_ci95_halfwidth=half.tolist(),
                    paired_absolute_drift_upper_bound_GPa=upper.tolist(),samples=count,
                    combined_ci95_halfwidth=combined.tolist(),passed=ok,elapsed_s=test['elapsed_s'])
    # Check common sampler candidates using only the training temperatures.
    # Stop at the first setting passing for every selected model.
    for steps,amp in [(25,True),(50,True),(50,False),(25,False)]:
        accepted=True;rows=[]
        for objective in p['models']['all_objectives']:
            step=selection[objective]['selected_step']
            for T in p['temperatures_K']['train']:
                # Same latent seeds: paired uncertainty isolates integration/precision drift.
                row=compare(objective,T,step,100,steps,amp)
                accepted=accepted and row['passed'];rows.append(row)
        candidates.append(dict(steps=steps,amp=amp,passed=accepted,rows=rows,total_seconds=sum(x['elapsed_s'] for x in rows)))
        write_json(path,dict(complete=False,candidates=candidates))
        if accepted:break
    valid=[x for x in candidates if x['passed']]
    convergence=[]
    if valid:
        best=min(valid,key=lambda x:x['total_seconds'])
    else:
        # The fallback requires an additional convergence check, not
        # silently treating 100 steps as accurate merely because it was first.
        best=None
        for coarse,fine in [(100,200),(200,400),(400,800)]:
            rows=[]
            for objective in p['models']['all_objectives']:
                step=selection[objective]['selected_step']
                for T in p['temperatures_K']['train']:
                    rows.append(compare(objective,T,step,fine,coarse,False))
            passed=all(row['passed'] for row in rows)
            convergence.append(dict(coarse_steps=coarse,fine_steps=fine,passed=passed,rows=rows))
            write_json(path,dict(complete=False,candidates=candidates,extended_convergence=convergence))
            if passed:
                best=dict(steps=coarse,amp=False);break
        if best is None:
            raise RuntimeError('Heun stress convergence remains unresolved through 800 steps; retain all evidence for review.')
    write_json(path,dict(complete=True,steps=best['steps'],amp=best['amp'],candidates=candidates,
                         extended_convergence=convergence,
                         note='Use the first passing common setting in the predeclared order (25 BF16, 50 BF16, 50 FP32, 25 FP32). Each absolute paired drift plus its pointwise 95% halfwidth must be <=0.01 GPa; the point drift must also lie within combined sampling intervals. Start with 2048 pairs, refine unresolved cases to 10000; fallback checks FP32 grid convergence.'))


def final_evaluation():
    p=protocol();sel=json.loads((ROOT/'results/checkpoint_selection/selection.json').read_text())
    sampler=json.loads((ROOT/'results/sampler_selection.json').read_text());result={}
    for T in p['temperatures_K']['final_evaluation']:
        methods={'large_md':summarize(paths_for('reference',T)),'small_md':summarize(paths_for('small',T))}
        for objective in p['models']['all_objectives']+['gaussian']:
            step=sel[objective]['selected_step'] if objective!='gaussian' else 0
            reports=[]
            for k in range(16):
                reports.append(generated(objective,step,T,5000,sampler['steps'],sampler['amp'],'final',seed=951227+7919*k))
                if (k+1)%4:continue
                mean=np.mean([x['mean'] for x in reports],axis=0)
                cov=sum(np.array(x['covariance_of_mean']) for x in reports)/len(reports)**2
                half=1.96*np.sqrt(np.diag(cov)).reshape(3,3)
                resolved=all(half[i,j]<=.02 for i,j in [(0,0),(0,1)])
                if resolved:break
            methods[objective]=dict(mean=mean.tolist(),ci95_halfwidth=half.tolist(),
                covariance_of_mean=cov.tolist(),checkpoint=step,samples=5000*len(reports),runs=reports,
                precision_target_met=resolved)
        result[str(T)]=methods
        write_json(ROOT/'results/final_statistics.json',dict(complete=False,temperatures=result,sampler=sampler,
            checkpoints={k:v['selected_step'] for k,v in sel.items()}))
    write_json(ROOT/'results/final_statistics.json',dict(complete=True,temperatures=result,sampler=sampler,
        checkpoints={k:v['selected_step'] for k,v in sel.items()},completed_at=now()))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['checkpoints','switch','sampler','final'])
    parser.add_argument('--objective',choices=protocol()['models']['all_objectives'])
    args=parser.parse_args()
    parallel=json.loads((ROOT/'checks/observable_parallel_validation.json').read_text())
    resume=json.loads((ROOT/'checks/sampling_resume_validation.json').read_text())
    uncertainty=json.loads((ROOT/'checks/checkpoint_uncertainty_validation.json').read_text())
    if not parallel.get('passed') or not resume.get('passed') or not uncertainty.get('passed'):
        raise RuntimeError('Observable parallelism, sampling-resume and score-uncertainty checks must pass before evaluation.')
    if (parallel['physics_sha256']!=sha256(ROOT/'src/physics.py') or
        parallel['observable_backend_sha256']!=sha256(ROOT/'src/observable_parallel.py') or
        resume.get('evaluation_sha256')!=sha256(Path(__file__)) or
        uncertainty.get('evaluation_sha256')!=sha256(Path(__file__)) or
        resume.get('observable_backend_sha256')!=parallel['observable_backend_sha256']):
        raise RuntimeError('Evaluation checks do not match current source; retain results and revalidate changed implementation.')
    if args.stage=='checkpoints':select_checkpoints([args.objective] if args.objective else None)
    else:{'switch':select_switch,'sampler':select_sampler,'final':final_evaluation}[args.stage]()
