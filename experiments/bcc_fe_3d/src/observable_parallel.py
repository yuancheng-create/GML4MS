"""Ordered CPU evaluation of independent configurations using unchanged physics."""
import atexit
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from physics import EAM,stress

_eam=None
_pool=None
_workers=None


def evaluate_one(arguments):
    global _eam
    if _eam is None:_eam=EAM()
    coordinates,F,a,temperature=arguments
    q=np.asarray(coordinates,dtype=np.float64)
    evaluation=_eam.evaluate(q)
    sigma=stress(q,evaluation,F,a,temperature)[0]
    return sigma,float(evaluation['distances'].min())


def close_pool():
    global _pool,_workers
    if _pool is not None:_pool.shutdown(wait=True,cancel_futures=True)
    _pool=None;_workers=None


atexit.register(close_pool)


def evaluate_observables(batch,F,a,temperature,workers=4):
    """Yield in input order; exceptions propagate without replacing any sample."""
    global _pool,_workers
    tasks=((q,F,a,temperature) for q in batch)
    if workers==1:return map(evaluate_one,tasks)
    if not isinstance(workers,int) or workers<1:raise ValueError('Invalid observable worker count')
    if _pool is None:
        # Never fork a process with an initialized CUDA context.
        _pool=ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn'))
        _workers=workers
    elif workers!=_workers:
        raise RuntimeError('Observable worker count changed within one evaluation process')
    return _pool.map(evaluate_one,tasks,chunksize=1)
