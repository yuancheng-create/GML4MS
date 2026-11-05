"""Manuscript-checkpoint, unfiltered sampling with matched latent random draws."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '2'
import argparse
import fcntl
import json
import sys
import time
import numpy as np
import torch
from common import TEMPS, PROTOCOL, case_root, atomic_json, digest, now
from preprocessing import gaussian_parameters
from network import build_model

from diffusion import sample_latent
from observables import measure_batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vacancies', type=int, choices=[7], required=True)
    p.add_argument('--method', choices=['force', 'noise', 'joint', 'gaussian'], required=True)
    p.add_argument('--seed', type=int, choices=[0], default=0)
    p.add_argument('--checkpoint-step', type=int, choices=[6000, 100000], default=100000)
    args = p.parse_args()
    if args.vacancies not in PROTOCOL['vacancy_counts'] or args.seed >= PROTOCOL['training_repetitions']:
        p.error('This workflow supports seven vacancies and seed 0 only')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = case_root(args.vacancies)
    geo = np.load(root/'geometry.npz')
    atoms = len(geo['qref']); dim = 2*atoms
    params = np.load(root/'processed'/'gaussian_parameters.npz')
    parameter_hash = digest(root/'processed'/'gaussian_parameters.npz')
    model = None
    model_hash = None
    if args.method != 'gaussian':
        checkpoint_name = 'final.pt' if args.checkpoint_step == 100000 else 'step_006000.pt'
        checkpoint = root/'models'/f'{args.method}_seed{args.seed}'/checkpoint_name
        cp = torch.load(checkpoint, map_location='cuda', weights_only=False)
        if cp['step'] != args.checkpoint_step:
            raise ValueError('Checkpoint step does not match the requested comparison')
        model = build_model(cp['config']).cuda().to(memory_format=torch.channels_last)
        for key in ['means', 'cholesky', 'reference', 'indices']:
            if not torch.equal(getattr(model, key), cp['ema'][key]):
                raise ValueError(f'Changed geometry or preprocessing: {key}')
        model.load_state_dict(cp['ema']); model.eval()
        model_hash = digest(checkpoint)
        validation_file=root/'models'/f'{args.method}_seed{args.seed}'/f'sampling_validation_step{args.checkpoint_step}.json'
        if not validation_file.exists():
            records=[]
            for temperature in [.02,.10]:
                mu,cov=gaussian_parameters(temperature,params)
                chol=np.linalg.cholesky(cov)
                seed=131026
                full=sample_latent(model,temperature,32,400,32,seed,amp=False,cuda_graph=False)
                captured=sample_latent(model,temperature,32,400,32,seed,amp=False,cuda_graph=True)
                selected=sample_latent(model,temperature,32,25,32,seed,amp=True,cuda_graph=True)
                points=[(z.astype(np.float64)@chol.T+mu).reshape(-1,atoms,2) for z in [full,captured,selected]]
                sm=[measure_batch(q,geo['qref'],temperature,geo['pairs'],geo['selected'],geo['core_tags'])[0] for q in points]
                coordinate_error=float(np.sqrt(np.mean((points[0]-points[1])**2)))
                if coordinate_error>1e-7 or not all(np.isfinite(q).all() for q in points):
                    raise ValueError('Float32 CUDA-graph implementation check failed')
                records.append(dict(temperature=temperature,paired_samples=32,
                    captured_vs_plain_float32_coordinate_rms=coordinate_error,
                    captured_vs_plain_float32_mean_stress_difference=float((sm[1][:,3]-sm[0][:,3]).mean()),
                    bf16_steps25_vs_fp32_steps400_mean_stress_difference=float((sm[2][:,3]-sm[1][:,3]).mean())))
            atomic_json(validation_file,dict(passed=True,checked_at=now(),checkpoint_sha256=model_hash,records=records,
                scope='paired numerical implementation checks; not additional production samples or distribution validation'))
    output = root/'generated' if args.checkpoint_step == 100000 else root/'generated_step6000'
    for temp in TEMPS:
        folder = output/f'{args.method}_seed{args.seed}_T{temp:.3f}'
        folder.mkdir(parents=True, exist_ok=True)
        lock = (folder/'.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = dict(vacancy_count=args.vacancies, method=args.method, seed=args.seed,
                      temperature=temp, count=50000, integration_steps=25 if model else 0,
                      batch=1000, chunk=2000, atoms=atoms, checkpoint_sha256=model_hash,
                      gaussian_parameters_sha256=parameter_hash,
                      precision='BF16 network, FP32 ODE without TF32; FP64 coordinates and physical statistics',
                      rng='CUDA generator; same seed/chunks across neural methods and Gaussian seed0')
        if (folder/'meta.json').exists():
            meta = json.loads((folder/'meta.json').read_text())
            if meta['configuration'] != config:
                raise ValueError('Changed committed sampling condition')
            print('exists', folder.name, flush=True)
            continue
        wall = time.monotonic()
        if (folder/'progress.json').exists():
            progress = json.loads((folder/'progress.json').read_text())
            if progress['configuration'] != config:
                raise ValueError('Changed resumed sampling condition')
            completed, previous = progress['completed_samples'], progress['seconds']
            coords = np.load(folder/'coordinates.npy', mmap_mode='r+')
            stats = np.load(folder/'statistics.npy', mmap_mode='r+')
            hist = np.load(folder/'histograms.npy', mmap_mode='r+')
        else:
            completed, previous = 0, 0.
            coords = np.lib.format.open_memmap(folder/'coordinates.npy', mode='w+', dtype=np.float64, shape=(50000, atoms, 2))
            stats = np.lib.format.open_memmap(folder/'statistics.npy', mode='w+', dtype=np.float64, shape=(50000, 18))
            hist = np.lib.format.open_memmap(folder/'histograms.npy', mode='w+', dtype=np.int64, shape=(250, 3, 680))
        mu, cov = gaussian_parameters(temp, params)
        chol = np.linalg.cholesky(cov)
        for start in range(completed, 50000, 2000):
            seed = 948222 + 10000*args.seed + int(round(temp*1000)) + (start//2000)*1000003
            if model is None:
                rng = torch.Generator(device='cuda').manual_seed(seed)
                latent = torch.cat([torch.randn((1000, dim), device='cuda', generator=rng).cpu() for _ in range(2)]).numpy()
            else:
                latent = sample_latent(model, temp, 2000, 25, 1000, seed, amp=True, cuda_graph=True)
            positions = (latent.astype(np.float64)@chol.T+mu).reshape(-1, atoms, 2)
            if not np.isfinite(positions).all():
                raise ValueError('Nonfinite generated coordinates; no sample filtering is permitted')
            sm, bh = measure_batch(positions, geo['qref'], temp, geo['pairs'], geo['selected'], geo['core_tags'])
            if not np.isfinite(sm[:, :16]).all():
                raise ValueError('Nonfinite generated physical statistics')
            coords[start:start+2000] = positions
            stats[start:start+2000] = sm
            hist[start//200:start//200+10] = bh
            coords.flush(); stats.flush(); hist.flush()
            atomic_json(folder/'progress.json', dict(configuration=config, completed_samples=start+2000,
                        seconds=previous+time.monotonic()-wall))
            print(json.dumps(dict(case=folder.name, completed=start+2000)), flush=True)
        atomic_json(folder/'meta.json', dict(configuration=config, count=50000, method=args.method,
            seed=args.seed, temperature=temp, completed_at=now(), seconds=previous+time.monotonic()-wall,
            checkpoint_step=args.checkpoint_step if model else None, rect_stress=float(stats[:, 3].mean()),
            min_pair_distance=float(stats[:, 12].min()), max_displacement=float(stats[:, 13].max())))


if __name__ == '__main__':
    main()
