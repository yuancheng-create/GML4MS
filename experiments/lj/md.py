"""Independent LJ MD tasks and pre-production seven-vacancy stability gate."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from common import ROOT, A, TRAIN_TEMPS, TEMPS, atomic_json, case_root, now

from md_data import run_md
from observables import candidates, STAT_NAMES


def one(task):
    vacancies, n, temperature, split, replica, steps, equil = task
    root = case_root(vacancies)
    dest = root / ('pilot' if split == 'pilot' else 'data') / f'n{n}_T{temperature:.3f}_{split}_r{replica}'
    configuration = dict(vacancy_count=vacancies, n=n, temperature=temperature,
                         split=split, replica=replica, production_steps=steps,
                         equilibration_steps=equil, save_stride=100, dt=.002, gamma=1.)
    if (dest / 'meta.json').exists():
        meta = json.loads((dest / 'meta.json').read_text())
        if any(meta[k] != v for k, v in configuration.items()):
            raise ValueError(f'Changed MD configuration: {dest}')
        return meta
    dest.mkdir(parents=True, exist_ok=True)
    z = np.load(root / 'initial' / f'n{n}.npz')
    q = z['q'].copy()
    if split == 'train_extension':
        q = np.load(root / 'data' / f'n{n}_T{temperature:.3f}_train_r{replica}' / 'observables.npz')['qlast']
    pref = z['patchref']
    pairs = candidates(pref) if n >= 18 else np.array([(i, j) for i in range(len(pref)) for j in range(i+1, len(pref))], dtype=np.int64)
    code = {'pilot': 0, 'train': 1, 'validation': 2, 'test': 3,
            'small': 4, 'train_extension': 6}[split]
    seed = 13102026 + 1000000*vacancies + 10000*n + 100000*code + 1000*replica + int(temperature*1000)
    start = time.monotonic()
    coords, sf, stats, hist, qlast, vlast = run_md(q, z['qref'], z['box'], z['ids'], pref,
            z['selected'], z['core_tags'], pairs, temperature, seed, equil, steps)
    if not np.isfinite(coords).all() or not np.isfinite(stats).all():
        raise ValueError(f'Nonfinite MD output: {dest}')
    np.save(dest / 'coordinates.npy', coords)
    if split in ['train', 'train_extension', 'validation']:
        np.save(dest / 'forces.npy', sf)
    np.savez_compressed(dest / 'observables.npz', stats=stats, hist=hist, qref=pref,
                        selected=z['selected'], core_tags=z['core_tags'], box=z['box'],
                        qlast=qlast, vlast=vlast)
    aligned_last = qlast - (qlast-z['qref']).mean(axis=0)
    meta = dict(**configuration, seed=seed, frames=len(coords), full_atoms=len(z['qref']),
                retained_atoms=len(pref), selected_atoms=int(z['selected'].sum()),
                columns=STAT_NAMES, seconds=time.monotonic()-start,
                max_patch_displacement=float(stats[:, 13].max()),
                last_full_displacement=float(np.linalg.norm(aligned_last-z['qref'], axis=1).max()),
                kinetic_temperature=float(stats[:, 16].mean()),
                min_distance=float(stats[:, 12].min()), completed_at=now())
    atomic_json(dest / 'meta.json', meta)
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--production', action='store_true')
    p.add_argument('--extend', action='store_true')
    p.add_argument('--workers', type=int, default=10)
    args = p.parse_args()
    if args.pilot:
        tasks = [(7, n, t, 'pilot', r, 200000, 100000)
                 for n in [8, 48] for t in [.02, .10] for r in range(2)]
    elif args.production:
        gate = json.loads((case_root(7) / 'stability.json').read_text())
        if not gate['passed']:
            raise RuntimeError('Seven-vacancy stability gate failed; no formal production authorized for an unstable state.')
        tasks = [(7, 48, t, s, r, 1000000, 100000) for t in TRAIN_TEMPS
                 for s, reps in [('train', 6), ('validation', 2), ('test', 4)] for r in range(reps)]
        tasks += [(7, 48, t, 'test', r, 1000000, 100000) for t in TEMPS if t not in TRAIN_TEMPS for r in range(4)]
        tasks += [(7, 8, t, 'small', r, 1000000, 100000) for t in TEMPS for r in range(4)]
    elif args.extend:
        tasks = [(7, 48, t, 'train_extension', r, 1000000, 100000) for t in TRAIN_TEMPS for r in range(6)]
    else:
        p.error('select an explicit MD phase')
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(one, task) for task in tasks]):
            result = future.result()
            rows.append(result)
            print(json.dumps(result), flush=True)
    if args.pilot:
        threshold = .45*A
        # This is a representation/stability diagnostic, not a sample rejection rule.
        passed = all(row['max_patch_displacement'] < threshold and
                     row['last_full_displacement'] < threshold and
                     abs(row['kinetic_temperature']/row['temperature']-1) < .03 for row in rows)
        atomic_json(case_root(7) / 'stability.json', dict(passed=passed, checked_at=now(),
                criterion='all recorded patch displacements and terminal full-system displacements below 0.45a; temperature within 3%',
                displacement_threshold=threshold, records=rows,
                limitation='short trajectories diagnose a stable labelled basin, not global equilibrium or absence of rare transitions'))
        if not passed:
            raise RuntimeError('Seven-vacancy short-MD stability gate failed.')


if __name__ == '__main__':
    main()
