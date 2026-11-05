"""Matched initialization and random draws; three losses, fixed update budget."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '2'
import argparse
import copy
import fcntl
import hashlib
import json
import time
import numpy as np
import torch
from common import PROTOCOL, case_root, atomic_json, digest, now
from network import PhysicalScoreModel


def losses(prediction, noisy, force_label, noise, alpha, sigma):
    force = (prediction-force_label).square().mean()
    predicted_noise = sigma*noisy + alpha*sigma*prediction
    denoising = (predicted_noise-noise).square().mean()
    return force, denoising, .5*force+4*denoising


def validation(model, z, labels, temps):
    generator = torch.Generator(device='cuda').manual_seed(4482)
    result = np.zeros(3)
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            ix = torch.randint(len(z), (512,), device='cuda', generator=generator)
            tau = torch.rand(512, device='cuda', generator=generator)
            a = torch.cos(torch.pi*tau/2)[:, None]
            s = torch.sin(torch.pi*tau/2)[:, None]
            eps = torch.randn(z[ix].shape, device='cuda', generator=generator)
            noisy = a*z[ix]+s*eps
            pred = model(noisy, tau, temps[ix])
            result += np.array([float(v) for v in losses(pred, noisy, labels[ix], eps, a, s)])/20
    return dict(validation_force_loss=result[0], validation_noise_loss=result[1], validation_joint_loss=result[2])


def state_hash(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vacancies', type=int, choices=[7], required=True)
    parser.add_argument('--objective', choices=['force', 'noise', 'joint'], required=True)
    parser.add_argument('--seed', type=int, choices=[0], required=True)
    args = parser.parse_args()
    if args.vacancies not in PROTOCOL['vacancy_counts'] or args.seed >= PROTOCOL['training_repetitions']:
        parser.error('This workflow supports seven vacancies and seed 0 only')
    root = case_root(args.vacancies)
    folder = root/'models'/f'{args.objective}_seed{args.seed}'
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder/'.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder/'training_complete.json').exists():
        done = json.loads((folder/'training_complete.json').read_text())
        if done['steps'] != 100000 or done['checkpoint_sha256'] != digest(folder/'final.pt'):
            raise ValueError('Existing final model is inconsistent')
        print('exists', folder, flush=True)
        return
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(8921+args.seed)
    model = PhysicalScoreModel(case_path=root).cuda().to(memory_format=torch.channels_last)
    initial_hash = state_hash(model)
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    generator = torch.Generator(device='cuda').manual_seed(731337+args.seed)
    processed = root/'processed'
    x = torch.from_numpy(np.load(processed/'train_z.npy')).cuda()
    labels = torch.from_numpy(np.load(processed/'train_residual.npy')).cuda()
    temps = torch.from_numpy(np.load(processed/'train_t.npy')).cuda()
    vz = torch.from_numpy(np.load(processed/'validation_z.npy')).cuda()
    vf = torch.from_numpy(np.load(processed/'validation_residual.npy')).cuda()
    vt = torch.from_numpy(np.load(processed/'validation_t.npy')).cuda()
    if not bool(torch.isfinite(vf).all()):
        raise ValueError('Missing validation force labels')
    start = 0
    history = []
    previous_seconds = 0.
    if (folder/'last.pt').exists():
        cp = torch.load(folder/'last.pt', map_location='cuda', weights_only=False)
        if cp['protocol'] != PROTOCOL or cp['initial_state_sha256'] != initial_hash:
            raise ValueError('Changed training setup at resume')
        model.load_state_dict(cp['model']); ema.load_state_dict(cp['ema'])
        optimizer.load_state_dict(cp['optimizer']); generator.set_state(cp['training_rng'].cpu())
        start = cp['step']
        history = json.loads((folder/'history.json').read_text())
        previous_seconds = history[-1]['total_seconds']
    wall = time.monotonic()
    first_draw_hash = None
    for step in range(start+1, 100001):
        model.train()
        ix = torch.randint(len(x), (256,), device='cuda', generator=generator)
        tau = torch.rand(256, device='cuda', generator=generator)
        a = torch.cos(torch.pi*tau/2)[:, None]
        s = torch.sin(torch.pi*tau/2)[:, None]
        eps = torch.randn((256, x.shape[1]), device='cuda', generator=generator)
        if step == 1:
            h = hashlib.sha256()
            for array in [ix, tau, eps]:
                h.update(array.cpu().numpy().tobytes())
            first_draw_hash = h.hexdigest()
            atomic_json(folder/'initialization.json', dict(initial_state_sha256=initial_hash,
                first_training_draw_sha256=first_draw_hash, seed=args.seed,
                objective=args.objective, train_frames=len(x), coordinate_dimension=x.shape[1]))
        noisy = a*x[ix]+s*eps
        pred = model(noisy, tau, temps[ix])
        f, n, j = losses(pred, noisy, labels[ix], eps, a, s)
        loss = {'force': f, 'noise': n, 'joint': j}[args.objective]
        if not bool(torch.isfinite(loss)):
            raise ValueError(f'Nonfinite training loss at {step}')
        lr = 2e-4*(.1+.9*(1+np.cos(np.pi*step/100000))/2)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
        with torch.no_grad():
            for averaged, current in zip(ema.parameters(), model.parameters()):
                averaged.lerp_(current, .001)
        if step % 1000 == 0:
            row = dict(step=step, train_force_loss=float(f), train_noise_loss=float(n),
                       train_joint_loss=float(j), total_seconds=previous_seconds+time.monotonic()-wall,
                       **validation(ema, vz, vf, vt))
            history.append(row)
            state = dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                config=model.config, step=step, seed=args.seed, objective=args.objective,
                initial_state_sha256=initial_hash, protocol=PROTOCOL,
                training_rng=generator.get_state())
            temporary = folder/'last.pending.pt'
            torch.save(state, temporary); temporary.replace(folder/'last.pt')
            if step == 6000:
                torch.save(state, folder/'step_006000.pt')
            atomic_json(folder/'history.json', history)
            print(json.dumps(dict(vacancies=args.vacancies, objective=args.objective, seed=args.seed, **row)), flush=True)
    os.link(folder/'last.pt', folder/'final.pt')
    atomic_json(folder/'training_complete.json', dict(completed_at=now(), steps=100000,
        objective=args.objective, seed=args.seed, initial_state_sha256=initial_hash,
        checkpoint_sha256=digest(folder/'final.pt'), selection_rule=PROTOCOL['checkpoint_rule']))


if __name__ == '__main__':
    main()
