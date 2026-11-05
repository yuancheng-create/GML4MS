"""Compare ordered parallel observables against the unchanged serial evaluator."""
import json
import time
import numpy as np
from common import ROOT,now,protocol,sha256,write_json
from observable_parallel import evaluate_observables,close_pool


def main():
    p=protocol();F=np.array(p['geometry']['F']);a=p['potential']['a0_verified_A'];rows=[]
    workers=p['generation']['observable_workers']
    for T in p['temperatures_K']['train']:
        path=sorted((ROOT/'data').glob(f'validation_n40_T{T}_*/coordinates.npy'))[0]
        saved=np.load(path,mmap_mode='r');batch=np.array(saved[np.linspace(0,len(saved)-1,32,dtype=int)])
        # Warm-up costs are recorded separately; these concurrent-run timings
        # serve scheduling only and are not a sampler performance experiment.
        start=time.monotonic();serial=list(evaluate_observables(batch,F,a,T,workers=1));serial_s=time.monotonic()-start
        start=time.monotonic();parallel=list(evaluate_observables(batch,F,a,T,workers=workers));parallel_s=time.monotonic()-start
        equal=all(np.array_equal(x[0],y[0]) and x[1]==y[1] for x,y in zip(serial,parallel))
        assert len(serial)==len(parallel)==len(batch) and equal
        rows.append(dict(temperature_K=T,source=str(path.relative_to(ROOT)),samples=len(batch),
            ordered_results_bitwise_equal=equal,serial_seconds=serial_s,parallel_seconds=parallel_s))
    # A known invalid input must surface at its own position in the ordered
    # iterator, rather than dropping it or attributing it to another sample.
    bad=[np.array(q) for q in batch[:8]];bad[3][0,0]=np.nan
    count=0;propagated=False
    try:
        for result in evaluate_observables(bad,F,a,T,workers=workers):count+=1
    except ValueError:
        propagated=True
    assert propagated and count==3
    close_pool()
    report=dict(at=now(),passed=True,workers=workers,scope='Implementation check using existing validation configurations only; no new trajectories or generated samples.',
        physics_sha256=sha256(ROOT/'src/physics.py'),observable_backend_sha256=sha256(ROOT/'src/observable_parallel.py'),
        cases=rows,ordered_error_propagation=True,
        timing_note='Implementation timings include process startup and do not establish controlled performance.')
    write_json(ROOT/'checks/observable_parallel_validation.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
