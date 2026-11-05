"""Train, select, and generate from locally regenerated MD; no report dependencies."""
import argparse
import fcntl
import json
import subprocess
import sys
import numpy as np
from common import ROOT, protocol, sha256, write_json, now
from learning import paths_for


def child(name, *args):
    print('Running:', name, *args, flush=True)
    subprocess.run([sys.executable, '-u', '-B', str(ROOT/'src'/name), *args], check=True, cwd=ROOT)


def check_data():
    p = protocol()
    acceptance = json.loads((ROOT/'checks/new_md_complete.json').read_text())
    if not acceptance['complete'] or acceptance['protocol_sha256'] != sha256(ROOT/'src/settings.py'):
        raise RuntimeError('MD campaign is incomplete or uses a different protocol.')
    seen_seeds = set()
    records = []
    for T in p['temperatures_K']['final_evaluation']:
        roles = ['train', 'validation', 'reference', 'small'] if T in p['temperatures_K']['train'] else ['reference', 'small']
        for role in roles:
            paths = paths_for(role, T)
            expected = 2 if role == 'validation' else 4
            if len(paths) != expected:
                raise RuntimeError(f'Expected {expected} independent {role} chains at {T} K.')
            for folder in paths:
                meta = json.loads((folder/'meta.json').read_text())
                seed = meta['seed']
                if seed in seen_seeds:
                    raise RuntimeError('MD roles must have independent seeds.')
                seen_seeds.add(seed)
                if meta.get('failure') or meta['potential_sha256'] != p['potential']['sha256']:
                    raise RuntimeError('MD failure or changed potential input.')
                x = np.load(folder/'coordinates.npy', mmap_mode='r')
                f = np.load(folder/'forces.npy', mmap_mode='r')
                if x.shape != f.shape or len(x) != meta['frames_written'] or len(x) != meta['frames']:
                    raise RuntimeError('Incomplete coordinates/force labels.')
                if role in ['train', 'validation'] and len(x) != 2000:
                    raise RuntimeError('The frozen training protocol requires 2000 frames per chain.')
                records.append(dict(folder=str(folder.relative_to(ROOT)), role=role, temperature_K=T, seed=seed))
    write_json(ROOT/'checks/data_isolation.json', dict(passed=True, at=now(), trajectories=records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['train', 'generate', 'all'], default='all')
    args = parser.parse_args()
    protocol()
    lock = (ROOT/'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    check_data()
    if args.stage in ['train', 'all']:
        for name in ['check_temperature_scope.py', 'check_learning_implementation.py',
                     'check_force_branch.py', 'check_sampling_resume.py',
                     'check_checkpoint_uncertainty.py', 'check_observable_parallel.py']:
            child(name)
        identities = {x.name: sha256(x) for x in (ROOT/'src').iterdir() if x.suffix in ['.py', '.cu']}
        identities['settings.py'] = sha256(ROOT/'src/settings.py')
        frozen = ROOT/'provenance/training_sources.json'
        if frozen.exists() and json.loads(frozen.read_text()) != identities:
            raise RuntimeError('Sources changed since training began; retain prior outputs and use a fresh checkout.')
        write_json(frozen, identities)
        child('learning.py', 'preprocess')
        child('learning.py', 'force')
        child('evaluation.py', 'checkpoints', '--objective', 'force')
        child('evaluation.py', 'switch')
        for objective in ['force_then_denoising', 'denoising']:
            child('learning.py', objective)
            child('evaluation.py', 'checkpoints', '--objective', objective)
    if args.stage in ['generate', 'all']:
        child('evaluation.py', 'sampler')
        child('evaluation.py', 'final')
    print('Requested stage complete. Outputs remain local and are ignored by Git.')


if __name__ == '__main__':
    main()
