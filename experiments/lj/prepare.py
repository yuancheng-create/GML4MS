"""Construct and relax the LJ seven-vacancy geometries."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
import argparse
import json
import numpy as np
from scipy.optimize import minimize
from common import ROOT, A, PROTOCOL, atomic_json, case_root, now
from md_core import lattice, neighbors, forces


def make_lattice(n, vacancies):
    full, box = lattice(n, A)
    if vacancies != 7:
        raise ValueError('This workflow supports the seven-vacancy system only.')
    remove = np.linalg.norm(full, axis=1) <= A + 1e-8
    if remove.sum() != vacancies:
        raise ValueError((n, vacancies, remove.sum()))
    qref = full[~remove]
    surface = np.any(np.linalg.norm(qref[:, None] - full[remove][None, :], axis=2) < 1.4*A, axis=1)
    return qref, box, surface


def objective(flat, box):
    q = flat.reshape(-1, 2)
    f, energy, _ = forces(q, box, neighbors(q, box))
    return energy, -f.ravel()


def prepare(vacancies):
    dest = case_root(vacancies)
    (dest / 'initial').mkdir(parents=True, exist_ok=True)
    records = []
    for n in [8, 48]:
        path = dest / 'initial' / f'n{n}.npz'
        if path.exists():
            continue
        qref, box, surface = make_lattice(n, vacancies)
        trials = []
        for perturb in [0., 1e-4*A]:
            initial = qref + perturb*np.random.default_rng(131026+n).normal(size=qref.shape)
            opt = minimize(objective, initial.ravel(), args=(box,), jac=True,
                           method='L-BFGS-B', options={'gtol': 1e-9, 'ftol': 1e-15,
                                                   'maxiter': 10000, 'maxls': 40})
            trials.append((opt.fun, opt.x.reshape(-1, 2), dict(energy=float(opt.fun),
                perturbation=perturb, optimizer_success=bool(opt.success), message=str(opt.message))))
        _, q, _ = min(trials, key=lambda t: t[0])
        trial_info = [t[2] for t in trials]
        q -= (q - qref).mean(axis=0)
        f, energy, _ = forces(q, box, neighbors(q, box))
        if n >= 18:
            half = np.array([9*A, 9*np.sqrt(3)*A/2])
            ids = np.flatnonzero(np.all((qref >= -half-1e-9) & (qref < half-1e-9), axis=1))
        else:
            ids = np.arange(len(qref))
        patchref = qref[ids]
        core = np.all((patchref >= -np.array([4*A, 2*np.sqrt(3)*A])-1e-9) &
                      (patchref < np.array([4*A, 2*np.sqrt(3)*A])-1e-9), axis=1)
        np.savez(path, q=q, qref=qref, box=box, ids=ids, patchref=patchref,
                 selected=surface[ids], core_tags=core, vacancy_count=vacancies)
        item = dict(n=n, vacancy_count=vacancies, atoms=len(q), retained_atoms=len(ids),
                    surface_atoms=int(surface.sum()), max_force=float(np.abs(f).max()),
                    max_relaxation=float(np.linalg.norm(q-qref, axis=1).max()),
                    energy=float(energy), trials=trial_info)
        records.append(item)
        print(json.dumps(item), flush=True)
        if n == 48:

            from observables import candidates
            np.savez(dest / 'geometry.npz', qref=patchref, relaxed=q[ids],
                     selected=surface[ids], core_tags=core, pairs=candidates(patchref),
                     full_atoms=len(q), retained_atoms=len(ids), vacancy_count=vacancies)
    if records:
        atomic_json(dest / 'relaxation.json', {'completed_at': now(), 'records': records})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vacancies', type=int, nargs='+', choices=[7], default=[7])
    args = parser.parse_args()
    atomic_json(ROOT / 'protocol.json', PROTOCOL)
    for v in args.vacancies:
        prepare(v)


if __name__ == '__main__':
    main()
