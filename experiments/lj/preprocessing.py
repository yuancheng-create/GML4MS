"""Training-only Gaussian whitening and geometry-aware full-system force labels."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '2'
import argparse
import json
import numpy as np
from scipy.linalg import solve_triangular
from common import TRAIN_TEMPS, case_root, atomic_json, digest, now


def gaussian_parameters(temp, parameters):
    ts = parameters['temperatures']
    if temp <= ts[0]:
        return parameters['means'][0], parameters['covariances'][0]
    if temp >= ts[-1]:
        return parameters['means'][-1], parameters['covariances'][-1]
    j = np.searchsorted(ts, temp)-1
    w = (temp-ts[j])/(ts[j+1]-ts[j])
    return ((1-w)*parameters['means'][j] + w*parameters['means'][j+1],
            temp*((1-w)*parameters['covariances'][j]/ts[j] + w*parameters['covariances'][j+1]/ts[j+1]))


def process(vacancies):
    root = case_root(vacancies)
    out = root/'processed'
    out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():
        return
    geo = np.load(root/'geometry.npz')
    atoms, full = int(geo['retained_atoms']), int(geo['full_atoms'])
    d = 2*atoms
    per = []
    path_groups = []
    for t in TRAIN_TEMPS:
        paths = [root/'data'/f'n48_T{t:.3f}_train_r{r}' for r in range(6)]
        extensions = [root/'data'/f'n48_T{t:.3f}_train_extension_r{r}' for r in range(6)]
        if any((p/'meta.json').exists() for p in extensions):
            if not all((p/'meta.json').exists() for p in extensions):
                raise ValueError('Partial extension set cannot enter preprocessing')
            paths += extensions
        if not all((p/'forces.npy').exists() for p in paths):
            raise ValueError('Training data incomplete')
        path_groups.append(paths)
        per.append(sum(json.loads((p/'meta.json').read_text())['frames'] for p in paths))
    if len(set(per)) != 1:
        raise ValueError('Unequal temperature counts')
    total = sum(per)
    arrays = {}
    for name, shape in [('train_z', (total, d)), ('train_residual', (total, d)),
                        ('train_t', (total,)), ('validation_z', (100000, d)),
                        ('validation_residual', (100000, d)), ('validation_t', (100000,))]:
        arrays[name] = np.lib.format.open_memmap(out/f'{name}.npy', mode='w+', dtype=np.float32, shape=shape)
    means, covs, chols, checks = [], [], [], []
    for index, (temp, paths) in enumerate(zip(TRAIN_TEMPS, path_groups)):
        x = np.concatenate([np.load(p/'coordinates.npy').reshape(-1, d) for p in paths])
        f = np.concatenate([np.load(p/'forces.npy') for p in paths])
        mu = x.mean(axis=0)
        delta = x-mu
        cov = delta.T@delta/len(x)
        cov += np.trace(cov)/d*1e-9*np.eye(d)
        chol = np.linalg.cholesky(cov)
        z = solve_triangular(chol, delta.T, lower=True).T
        corrected = f + f.sum(axis=1, keepdims=True)/(full-atoms)
        score = corrected.reshape(-1, d)@chol/temp
        sl = slice(index*per[index], (index+1)*per[index])
        arrays['train_z'][sl] = z
        arrays['train_residual'][sl] = -score-z
        arrays['train_t'][sl] = temp
        vp = [root/'data'/f'n48_T{temp:.3f}_validation_r{r}' for r in range(2)]
        vx = np.concatenate([np.load(p/'coordinates.npy').reshape(-1, d) for p in vp])
        vf = np.concatenate([np.load(p/'forces.npy') for p in vp])
        vz = solve_triangular(chol, (vx-mu).T, lower=True).T
        vs = (vf+vf.sum(axis=1, keepdims=True)/(full-atoms)).reshape(-1, d)@chol/temp
        sl = slice(index*20000, (index+1)*20000)
        arrays['validation_z'][sl] = vz
        arrays['validation_residual'][sl] = -vs-vz
        arrays['validation_t'][sl] = temp
        means.append(mu); covs.append(cov); chols.append(chol)
        check = dict(temperature=temp, frames=len(x),
                     whitened_mean_max=float(abs(z.mean(axis=0)).max()),
                     whitened_variance_error=float(abs((z*z).mean(axis=0)-1).max()),
                     force_score_stein_trace=float(-np.sum(score*z)/len(x)/d),
                     full_atoms=full, retained_atoms=atoms)
        checks.append(check)
        print(json.dumps(check), flush=True)
    for array in arrays.values():
        array.flush()
    np.savez(out/'gaussian_parameters.npz', temperatures=TRAIN_TEMPS,
             means=np.array(means), covariances=np.array(covs), cholesky=np.array(chols))
    atomic_json(out/'diagnostics.json', checks)
    atomic_json(out/'complete.json', dict(completed_at=now(), frames_per_temperature=per[0],
                full_atoms=full, retained_atoms=atoms, parameters_sha256=digest(out/'gaussian_parameters.npz')))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--vacancies', type=int, choices=[7], required=True)
    a = p.parse_args()
    process(a.vacancies)
