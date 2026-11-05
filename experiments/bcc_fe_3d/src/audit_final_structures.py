"""Screen completed final MD against the relaxed periodic lattice; retain all data."""
import argparse
import itertools
import json

import numpy as np
from scipy.spatial import cKDTree

from common import ROOT, now, protocol, sha256, write_json
from learning import paths_for


class PeriodicReference:
    def __init__(self, n, p):
        self.n = n
        path = ROOT / 'checks' / f'n{n}_shear{p["geometry"]["F"][0][1]:.3f}_relaxed.npz'
        with np.load(path) as data:
            self.q0 = data['q'].copy()
            self.sites = data['sites'].copy()
            self.F = data['F'].copy()
            self.a = float(data['a'])
        if not np.array_equal(self.F, p['geometry']['F']):
            raise RuntimeError('Relaxed lattice has a different deformation')
        self.inverse_F = np.linalg.inv(self.F)
        self.length = n * self.a
        self.geometry_sha256 = sha256(path)
        shifts = np.array(list(itertools.product([-1, 0, 1], repeat=3)))
        self.translations = (shifts * self.length) @ self.F.T
        canonical = self.fold(self.q0)
        images = (canonical[None] + self.translations[:, None]).reshape(-1, 3)
        self.tree = cKDTree(images)
        self.image_labels = np.tile(np.arange(len(self.q0)), len(shifts))
        self.ids = (np.flatnonzero(np.all((self.sites >= -8) & (self.sites < 8), axis=1))
                    if n > 16 else np.arange(len(self.q0)))
        self.own = self.q0[self.ids]
        self.guaranteed_radius = self.tree.query(self.fold(self.own), k=2)[0][:, 1] / 2
        self.near = np.linalg.norm(self.sites[self.ids] * self.a, axis=1) < 2 * self.a + p['potential']['cutoff_A']

    def fold(self, q):
        material = q @ self.inverse_F.T
        material = (material + self.length / 2) % self.length - self.length / 2
        return material @ self.F.T

    def labels(self, q):
        return self.image_labels[self.tree.query(self.fold(q))[1]]

    def implementation_check(self):
        ids = self.ids[np.linspace(0, len(self.ids) - 1, 8, dtype=int)]
        # Coordinate probes only: periodic-image equivalence, distinct-site
        # sensitivity, and positions inside the guaranteed nearest-site ball.
        probes = self.q0[ids]
        for shift in self.translations:
            if not np.array_equal(self.labels(probes + 2 * shift), ids):
                raise RuntimeError('Periodic-image labels are inconsistent')
        moved = self.q0[np.roll(ids, 1)]
        if not np.array_equal(self.labels(moved), np.roll(ids, 1)):
            raise RuntimeError('Distinct-site reassignment probe failed')
        radius = self.tree.query(self.fold(probes), k=2)[0][:, 1] / 2
        if not np.array_equal(self.labels(probes + radius[:, None] * np.array([.1, .1, .1])), ids):
            raise RuntimeError('Nearest-site guarantee probe failed')


def audit(temperatures, roles=('reference','small')):
    p = protocol()
    references = {}
    for temperature in temperatures:
        if temperature not in p['temperatures_K']['final_evaluation']:
            raise RuntimeError('Unapproved temperature')
        rows = []
        for role in roles:
            if role in ['train','validation'] and temperature not in p['temperatures_K']['train']:
                raise RuntimeError('Training/validation structural screening used a held-out temperature')
            folders = paths_for(role, temperature)
            expected=2 if role=='validation' else 4
            if len(folders) != expected:
                raise RuntimeError(f'Need {expected} completed {role} chains at {temperature} K')
            for folder in folders:
                meta_hash = sha256(folder / 'meta.json')
                meta = json.loads((folder / 'meta.json').read_text())
                n = meta['n']
                if n not in references:
                    references[n] = PeriodicReference(n, p)
                    references[n].implementation_check()
                ref = references[n]
                with np.load(folder / 'geometry.npz') as data:
                    if not np.array_equal(data['sites'], ref.sites[ref.ids]):
                        raise RuntimeError('Stored atom labels differ from the relaxed reference')
                x = np.load(folder / 'coordinates.npy', mmap_mode='r')
                if not meta['complete'] or meta['frames_written'] != len(x) or x.shape[1:] != ref.own.shape:
                    raise RuntimeError('Incomplete or incompatible trajectory')
                maximum = near_rms_sum = 0.
                minimum_radius = float('inf')
                candidate_count = max_reassigned = 0
                events = []
                for begin in range(0, len(x), 64):
                    q = np.asarray(x[begin:begin + 64], dtype=np.float64)
                    if not np.isfinite(q).all():
                        raise RuntimeError('Nonfinite saved coordinates')
                    displacement = np.linalg.norm(q - ref.own, axis=2)
                    maximum = max(maximum, float(displacement.max()))
                    near_rms_sum += float(np.sqrt(np.mean(displacement[:, ref.near]**2, axis=1)).sum())
                    material = q @ ref.inverse_F.T
                    material = (material + ref.length / 2) % ref.length - ref.length / 2
                    minimum_radius = min(minimum_radius, float(np.linalg.norm(material, axis=2).min()))
                    frame, atom = np.nonzero(displacement >= ref.guaranteed_radius)
                    candidate_count += len(frame)
                    if len(frame):
                        assigned = ref.labels(q[frame, atom])
                        changed = assigned != ref.ids[atom]
                        counts = np.bincount(frame[changed], minlength=len(q))
                        max_reassigned = max(max_reassigned, int(counts.max(initial=0)))
                        for f, i, j in zip(frame[changed], atom[changed], assigned[changed]):
                            events.append(dict(frame=int(begin + f), time_ps=float((begin + f + 1) * meta['save_interval_ps']),
                                retained_atom=int(i), original_full_site=int(ref.ids[i]), nearest_full_site=int(j),
                                unwrapped_displacement_A=float(displacement[f, i])))
                if sha256(folder / 'meta.json') != meta_hash:
                    raise RuntimeError('Trajectory changed during screening; retry when complete')
                rows.append(dict(folder=str(folder.relative_to(ROOT)), role=role, frames=len(x), atoms=len(ref.own),
                    relaxed_geometry_sha256=ref.geometry_sha256, metadata_sha256=meta_hash,
                    coordinates_sha256=sha256(folder / 'coordinates.npy'),
                    maximum_unwrapped_displacement_A=maximum, mean_near_void_rms_A=near_rms_sum / len(x),
                    minimum_material_radius_A=minimum_radius, voronoi_candidates_checked=candidate_count,
                    maximum_simultaneously_reassigned_sites=max_reassigned, site_reassignment_events=events))
                print(json.dumps(dict(temperature_K=temperature, role=role, seed=meta['seed'], frames=len(x),
                    maximum_unwrapped_displacement_A=maximum, reassignment_events=len(events))), flush=True)
        tag='training_structure' if set(roles)=={'train','validation'} else 'final_structure'
        write_json(ROOT / 'checks' / f'{tag}_T{temperature}.json', dict(at=now(), complete=True,roles=list(roles),
            temperature_K=temperature, implementation_probes_passed=True, frames_removed=0, checks=rows,
            auditor_sha256=sha256(ROOT / 'src/audit_final_structures.py'),
            scope='Every saved frame in the listed completed MD roles. The full relaxed lattice and periodic images define nearest-site labels. Displacements use the aligned unwrapped labeled trajectory. This screening does not prove absence of plasticity or events between saved frames, and does not assess generated samples.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('temperature', nargs='+', type=int)
    parser.add_argument('--roles',nargs='+',choices=['train','validation','reference','small'],default=['reference','small'])
    args=parser.parse_args();audit(args.temperature,args.roles)
