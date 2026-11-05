"""Independent MD campaign for the manuscript BCC Fe system."""
import argparse
import fcntl
import json
import os
import sys
from pathlib import Path
import subprocess
import traceback

import numpy as np
from scipy.stats import t as student_t
from common import ROOT, now, protocol, sha256, write_json
from learning import paths_for
from md_jobs import trajectory, summarize


def status(stage, **extra):
    row = dict(stage=stage, pid=os.getpid(), at=now(), **extra)
    write_json(ROOT/'md_status.json', row)
    with (ROOT/'logs/md_events.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def check_stationarity(paths):
    differences = []
    for folder in paths:
        x = np.load(folder/'statistics.npy', mmap_mode='r')[:, :9]
        half = len(x)//2
        differences.append(x[half:].mean(0)-x[:half].mean(0))
    differences = np.array(differences)
    mean = differences.mean(0)
    half = student_t.ppf(.975, len(paths)-1)*differences.std(0, ddof=1)/np.sqrt(len(paths))
    # Screening diagnostic, not a proof of equilibrium. A resolvable drift
    # exceeding the requested stress accuracy requires more MD or review.
    unresolved = np.maximum(abs(mean)-half, 0)[[0, 1]] > .02
    return dict(first_to_second_half_drift_GPa=mean.tolist(), ci95_halfwidth_GPa=half.tolist(),
                persistent_drift=bool(np.any(unresolved)))


def generate(role, T, count, base, duration=200):
    n = 6 if role == 'small' else 40
    for chain in range(count):
        seed = base+T*101+chain*7919
        status('generating_MD', role=role, temperature_K=T, chain=chain, production_ps=duration)
        trajectory(n, T, role, seed, production_ps=duration, equilibration_ps=50, dt=.001)


def structure(T, roles):
    # This runs in a child so the large periodic nearest-site tree is released.
    command=['bash',str(ROOT/'run_python.sh'),str(ROOT/'src/audit_final_structures.py'),str(T),'--roles',*roles]
    with (ROOT/'logs'/f'structure_{T}_{roles[0]}.out').open('a') as out:
        subprocess.run(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
    tag='training_structure' if set(roles)=={'train','validation'} else 'final_structure'
    report=json.loads((ROOT/'checks'/f'{tag}_T{T}.json').read_text())
    if any(row['site_reassignment_events'] for row in report['checks']):
        raise RuntimeError(f'{T} K nearest-site reassignment detected; retain all frames and review structure before proceeding.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--temperatures',nargs='+',type=int,
        help='Run only these manuscript temperatures; never mark the whole campaign complete while others remain pending.')
    args=parser.parse_args()
    protocol()
    os.environ['SAM_PYTHON']=sys.executable
    order=protocol()['temperatures_K']['train']
    if args.temperatures is not None and (len(set(args.temperatures))!=len(args.temperatures)
            or not set(args.temperatures).issubset(order)):
        parser.error('--temperatures must be a nonrepeating subset of manuscript training temperatures')
    temperatures=[T for T in order if args.temperatures is None or T in args.temperatures]
    if args.detach:
        command=['bash',str(ROOT/'run_python.sh'),str(Path(__file__).resolve())]
        if args.temperatures is not None:command+=['--temperatures',*[str(T) for T in temperatures]]
        with (ROOT/'logs/md.out').open('a') as stream:
            process=subprocess.Popen(command,
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        fields=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()
        row=dict(pid=process.pid,process_start_ticks=int(fields[19]),at=now(),log=str(ROOT/'logs/md.out'),
            selected_temperatures_K=temperatures)
        write_json(ROOT/'logs/md_launcher.json',row);print(json.dumps(row));return
    lock=(ROOT/'md.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=protocol()
    try:
        status('checking_preparation')
        for name in ['physics_validation', 'gpu_backend_validation']:
            if not json.loads((ROOT/'checks'/f'{name}.json').read_text()).get('passed'):
                raise RuntimeError('Run src/prepare.py before MD.')
        prepared=json.loads((ROOT/'checks/preparation.json').read_text())
        if not prepared.get('passed') or prepared['protocol_sha256']!=sha256(ROOT/'src/settings.py'):
            raise RuntimeError('Preparation does not match the current protocol.')
        if any(sha256(ROOT/'src'/name)!=value for name,value in prepared['source_sha256'].items()):
            raise RuntimeError('Physical/backend sources changed; repeat preparation before MD.')
        sources=['common.py','physics.py','backend.py','gpu_md.py','eam_cuda.cu','md_jobs.py','run_md.py','audit_final_structures.py']
        launch=dict(at=now(),pid=os.getpid(),selected_temperatures_K=temperatures,
            protocol_sha256=sha256(ROOT/'src/settings.py'),
            source_sha256={name:sha256(ROOT/'src'/name) for name in sources})
        history=ROOT/'provenance/md_launch_history';history.mkdir(exist_ok=True)
        write_json(history/f'{os.getpid()}_{launch["at"].replace(":","-")}.json',launch)
        if not (ROOT/'provenance/md_launch.json').exists():
            write_json(ROOT/'provenance/md_launch.json',launch)
        # Test both endpoints first; pilot chains are retained as training data.
        for T in [T for T in protocol()['preproduction_checks']['pilot_temperatures_K'] if T in temperatures]:generate('train',T,1,1000000)
        for T in temperatures:
            marker=ROOT/'checks'/f'md_temperature_T{T}.json'
            if marker.exists():continue
            generate('train',T,4,1000000)
            generate('validation',T,2,2000000)
            structure(T,['train','validation'])
            for role in ['train','validation']:
                diagnostic=check_stationarity(paths_for(role,T))
                write_json(ROOT/'checks'/f'{role}_stationarity_T{T}.json',diagnostic)
                if diagnostic['persistent_drift']:
                    raise RuntimeError(f'{T} K {role} drift exceeds .02 GPa after uncertainty; review before training.')
            for role,base in [('reference',3000000),('small',4000000)]:
                resolved=False
                for duration in [200,400,800]:
                    generate(role,T,4,base,duration)
                    folders=paths_for(role,T);summary=summarize(folders)
                    drift=check_stationarity(folders)
                    write_json(ROOT/'results'/f'{role}_T{T}_summary.json',summary)
                    half=np.array(summary['ci95_halfwidth'])
                    resolved=bool(max(half[0,0],half[0,1])<=.02 and not drift['persistent_drift'])
                    write_json(ROOT/'checks'/f'{role}_convergence_T{T}.json',dict(at=now(),production_ps=duration,
                        precision_met=bool(max(half[0,0],half[0,1])<=.02),stationarity=drift,passed=resolved))
                    if resolved:break
                if not resolved:
                    raise RuntimeError(f'{role} at {T} K unresolved at 800 ps/chain; preserve data and request review.')
            structure(T,['reference','small'])
            write_json(marker,dict(complete=True,at=now(),temperature_K=T,structure_screen_passed=True))
        pending=[T for T in order if not (ROOT/'checks'/f'md_temperature_T{T}.json').exists()
            or not json.loads((ROOT/'checks'/f'md_temperature_T{T}.json').read_text()).get('complete')]
        if pending:
            status('selected_temperatures_complete',selected_temperatures_K=temperatures,
                pending_temperatures_K=pending)
            return
        # Generate independent held-out references from scratch; never use them for fitting.
        for T in p['temperatures_K']['held_out']:
            for role,base in [('reference',3000000),('small',4000000)]:
                resolved=False
                for duration in [200,400,800]:
                    generate(role,T,4,base,duration)
                    summary=summarize(paths_for(role,T));drift=check_stationarity(paths_for(role,T))
                    write_json(ROOT/'results'/f'{role}_T{T}_summary.json',summary)
                    half=np.array(summary['ci95_halfwidth'])
                    resolved=bool(max(half[0,0],half[0,1])<=.02 and not drift['persistent_drift'])
                    write_json(ROOT/'checks'/f'{role}_convergence_T{T}.json',dict(at=now(),production_ps=duration,
                        precision_met=bool(max(half[0,0],half[0,1])<=.02),stationarity=drift,passed=resolved))
                    if resolved:break
                if not resolved:raise RuntimeError(f'{role} at {T} K unresolved at 800 ps/chain; preserve data and review.')
            structure(T,['reference','small'])
        write_json(ROOT/'checks/new_md_complete.json',dict(complete=True,at=now(),
            generated_trajectories=14*len(order)+8*len(p['temperatures_K']['held_out']),
            train_temperatures=p['temperatures_K']['train'],protocol_sha256=sha256(ROOT/'src/settings.py')))
        status('md_complete')
    except Exception:
        status('stopped_with_error',traceback=traceback.format_exc())
        raise


if __name__=='__main__':main()
