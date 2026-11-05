"""Download the EAM input, validate physics, and prepare relaxed initial lattices."""
import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request

from common import ROOT, POTENTIAL, protocol, write_json, now, sha256


def fetch_potential():
    p = protocol()['potential']
    if POTENTIAL.exists():
        if sha256(POTENTIAL) != p['sha256']:
            raise RuntimeError('Existing EAM input has the wrong SHA256; inspect it before continuing.')
        return
    request = urllib.request.Request(p['source'], headers={'User-Agent': 'SAM-source-release'})
    with urllib.request.urlopen(request, timeout=120) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != p['sha256']:
        raise RuntimeError('Downloaded EAM input failed SHA256 verification; no file was installed.')
    pending = POTENTIAL.with_suffix('.pending')
    pending.write_bytes(content)
    pending.replace(POTENTIAL)


def validate_gpu():
    import numpy as np
    from gpu_md import GPUSimulation
    from physics import EAM
    geo = np.load(ROOT/'checks/n6_shear0.050_relaxed.npz')
    q, F, a = geo['q'], geo['F'], float(geo['a'])
    sim = GPUSimulation(q, F, 6*a, 50, 815337)
    eam = EAM()
    first = eam.evaluate(q, F, 6*a)
    force_error = float(abs(sim.forces()-first['forces']).max())
    energy_error = float(abs(sim.energy.sum().get()-first['energy']))
    sim.run(64)
    last = eam.evaluate(sim.positions(), F, 6*a)
    dynamic_error = float(abs(sim.forces()-last['forces']).max())
    passed = force_error < 1e-8 and dynamic_error < 1e-8 and energy_error < 1e-7
    write_json(ROOT/'checks/gpu_backend_validation.json', dict(
        at=now(), passed=passed, force_max_error_eV_per_A=force_error,
        energy_error_eV=energy_error, dynamic_force_max_error_eV_per_A=dynamic_error,
        backend_sources=sim.source_hashes, potential_sha256=sha256(POTENTIAL),
        scope='Small-cell CUDA parity against the independent EAM evaluator, itself checked against LAMMPS.'))
    if not passed:
        raise RuntimeError('GPU backend validation failed; production must not start.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-only', action='store_true', help='Fetch and verify the EAM input only.')
    args = parser.parse_args()
    protocol()
    fetch_potential()
    if args.download_only:
        print(POTENTIAL)
        return
    subprocess.run([sys.executable, '-B', str(ROOT/'src/validate_physics.py')], check=True)
    from md_jobs import geometry
    for n in [6, 40]:
        geometry(n, protocol()['geometry']['F'][0][1])
    validate_gpu()
    sources = ['physics.py', 'backend.py', 'gpu_md.py', 'eam_cuda.cu', 'md_jobs.py']
    write_json(ROOT/'checks/preparation.json', dict(passed=True, at=now(),
        protocol_sha256=sha256(ROOT/'src/settings.py'), potential_sha256=sha256(POTENTIAL),
        source_sha256={name:sha256(ROOT/'src'/name) for name in sources}))
    print('Physics, initial lattices, and CUDA backend are ready. No production MD has been run.')


if __name__ == '__main__':
    main()
