"""200 K ensemble fields from saved coordinates only; no trajectory or NN replay."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
from common import ROOT, KB, EV_A3_GPA, protocol, sha256, write_json
from physics import EAM, stress

OUT = ROOT / 'results/mean_fields_T200'
TEMPERATURE = 200
_EAM = None


def geometry():
    geo = np.load(ROOT / 'checks/n40_shear0.050_relaxed.npz')
    keep = np.all((geo['sites'] >= -8) & (geo['sites'] < 8), axis=1)
    sites, q0, ids = geo['sites'][keep], geo['q'][keep], np.flatnonzero(keep)+1
    result = dict(sites=sites, q0=q0, atom_ids=ids, F=geo['F'], a=geo['a'])
    return result


def evaluate_chunk(args):
    global _EAM
    if _EAM is None:
        _EAM = EAM()
    qs, F, a = args
    volume = np.linalg.det(F)*(6*a)**3
    atomic_sum = np.zeros(qs.shape[1]); sigmas = []; counts = []; error = 0.
    for q in qs:
        q = np.asarray(q, dtype=np.float64)
        ev = _EAM.evaluate(q)
        sigma, count, weight = stress(q, ev, F, a, TEMPERATURE)
        # Products are taken frame by frame, before the ensemble average.
        pair = -weight*ev['pair_forces'][:, 0]*ev['dr'][:, 0]
        atomic = .5*(np.bincount(ev['pairs'][:, 0], weights=pair, minlength=len(q)) +
                     np.bincount(ev['pairs'][:, 1], weights=pair, minlength=len(q)))
        reconstructed = (atomic.sum()-KB*TEMPERATURE*count)/volume*EV_A3_GPA
        error = max(error, abs(reconstructed-sigma[0, 0]))
        assert np.isfinite(atomic).all() and error < 1e-10
        atomic_sum += atomic; sigmas.append(sigma[0, 0]); counts.append(count)
    return atomic_sum, np.array(sigmas), np.array(counts), error


def sources(method):
    if method == 'large_md':
        role = 'reference'
        paths = sorted((ROOT/'data').glob(f'{role}_n*_T200_*'))
        assert len(paths) == 4
        return paths
    final = json.loads((ROOT/'results/final_statistics.json').read_text())
    assert final['complete']
    return [ROOT/r['folder'] for r in final['temperatures']['200'][method]['runs']]


def compute(method, pool):
    md = method == 'large_md'
    geo = geometry(); F = geo['F']; a = float(geo['a'])
    thermal = KB*TEMPERATURE/(np.linalg.det(F)*(6*a)**3)*EV_A3_GPA
    records = []; arrays = []; source_paths = sources(method)
    for index, source in enumerate(source_paths):
        started = time.monotonic()
        output = OUT/'statistics'/f'{method}_{index:02d}.npz'
        metadata = output.with_suffix('.json')
        coordinate_path = source/'coordinates.npy'
        x = np.load(coordinate_path, mmap_mode='r')
        if md:
            original = np.load(source/'statistics.npy', mmap_mode='r')
            expected = original[:, 0]+thermal*original[:, 11]
            source_meta = json.loads((source/'meta.json').read_text())
            assert source_meta['complete'] and len(x) == source_meta['frames_written']
        else:
            original = np.load(source/'stress.npy', mmap_mode='r')
            expected = original[:, 0, 0]
            source_meta = json.loads((source/'summary.json').read_text())
            assert source_meta['complete'] and len(x) == source_meta['samples']
        identity = dict(source=str(source.relative_to(ROOT)), frames=len(x),
                        coordinates_sha256=sha256(coordinate_path),
                        statistics_sha256=sha256(source/('statistics.npy' if md else 'stress.npy')),
                        physics_sha256=sha256(ROOT/'src/physics.py'),
                        postprocess_sha256=sha256(__file__),
                        reference_sha256=sha256(ROOT/'checks/n40_shear0.050_relaxed.npz'))
        if metadata.exists():
            record = json.loads(metadata.read_text())
            assert record['identity'] == identity, 'Input changed; do not silently reuse fields'
            records.append(record); arrays.append(dict(np.load(output))); continue
        count = 0; sum_atomic = np.zeros(len(geo['sites']))
        max_reconstruction = 0.; max_stored = 0.; boundary_changes = 0
        for start in range(0, len(x), 128):
            batch = np.asarray(x[start:start+128], dtype=np.float64)
            tasks = [(batch[j:j+16], F, a) for j in range(0, len(batch), 16)]
            for atomic, sigmas, counts, error in pool.map(evaluate_chunk, tasks):
                if md:
                    boundary_changes += int(np.count_nonzero(counts != original[count:count+len(counts), 11]))
                    compare = sigmas+thermal*counts
                else:
                    compare = sigmas
                difference = float(np.max(np.abs(compare-expected[count:count+len(compare)])))
                assert difference < (2e-4 if md else 1e-8), (method, count, difference)
                sum_atomic += atomic; count += len(sigmas)
                max_stored = max(max_stored, difference); max_reconstruction = max(max_reconstruction, error)
            if count % 1024 == 0 or count == len(x):
                print(method, index, count, '/', len(x), 'seconds', round(time.monotonic()-started, 1), flush=True)
        assert count == len(x)
        result = dict(mean_Axx_eV=sum_atomic/count)
        np.savez_compressed(output, **result)
        record = dict(identity=identity, frames=count, elapsed_s=time.monotonic()-started,
                      endpoint_sum_max_error_GPa=max_reconstruction,
                      stored_stress_max_difference_GPa=max_stored,
                      stored_comparison='configurational xx' if md else 'total xx',
                      MD_FP32_boundary_count_changes=boundary_changes)
        write_json(metadata, record); records.append(record); arrays.append(result)
    weights = np.array([r['frames'] for r in records])
    assert np.all(weights == weights[0]), 'Group SEM requires equal-size independent groups'
    result = dict(geo)
    for key in ['mean_Axx_eV']:
        values = np.array([d[key] for d in arrays])
        result[key] = np.average(values, weights=weights, axis=0)
        result['sem_'+key] = values.std(0, ddof=1)/np.sqrt(len(values))
    # Every method shows one predetermined saved sample, with two mean color fields.
    first = np.load(source_paths[0]/'coordinates.npy', mmap_mode='r')
    frame = int(np.random.default_rng(20044073).integers(len(first)))
    q = np.asarray(first[frame], dtype=float)
    result['snapshot_q'] = q
    np.savez_compressed(OUT/f'{method}_fields.npz', **result)
    record = dict(method=method, temperature_K=200, frames=int(weights.sum()), groups=records,
                  snapshot_source=str((source_paths[0]/'coordinates.npy').relative_to(ROOT)),
                  snapshot_frame_zero_based=frame, snapshot_selection_seed=20044073,
                  displacement_reference='common deformed and relaxed n40 configuration',
                  observable='half of each signed configurational IK xx bond contribution assigned to each endpoint; exact C bond fraction',
                  averaging='framewise force-displacement products, then ensemble average',
                  periodic_replication=False, display_atoms=len(q), passed=True)
    write_json(OUT/f'{method}_summary.json', record)
    print('Completed', method, 'frames', int(weights.sum()), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods', nargs='+', choices=['large_md', 'denoising', 'gaussian'])
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    (OUT/'statistics').mkdir(parents=True, exist_ok=True)
    methods = args.methods or ['large_md', 'denoising', 'gaussian']
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for method in methods:
            compute(method, pool)


if __name__ == '__main__':
    main()
