"""BAOAB integration kernel shared by the seven-vacancy MD workflows."""
import numpy as np
from numba import njit
from md_core import neighbors,forces,SKIN
from observables import measure


@njit(cache=True)
def run_md(q0,ref,box,ids,patchref,sel,core,localpairs,temp,seed,equil,steps):
    dt=.002
    np.random.seed(seed)
    q=q0.copy();v=np.random.normal(0.,np.sqrt(temp),q.shape)
    pairs=neighbors(q,box);build=q.copy()
    f,ep,vir=forces(q,box,pairs)
    decay=np.exp(-dt);noise=np.sqrt(temp*(1-decay*decay))
    frames=steps//100
    coords=np.empty((frames,len(ids),2))
    savedf=np.empty_like(coords)
    stats=np.empty((frames,18))
    bhist=np.zeros(((frames+199)//200,3,680),dtype=np.int64)
    fi=0
    for st in range(equil+steps):
        for i in range(len(q)):
            for d in range(2):
                v[i,d]+=.5*dt*f[i,d]
                q[i,d]+=.5*dt*v[i,d]
                v[i,d]=decay*v[i,d]+noise*np.random.normal()
                q[i,d]+=.5*dt*v[i,d]
        if np.max(np.sum((q-build)**2,axis=1))>(SKIN/2)**2:
            pairs=neighbors(q,box);build=q.copy()
        f,ep,vir=forces(q,box,pairs)
        for i in range(len(q)):
            for d in range(2):
                v[i,d]+=.5*dt*f[i,d]
        if st>=equil and (st-equil+1)%100==0:
            center=np.sum(q-ref,axis=0)/len(q)
            patch=q[ids]-center
            coords[fi]=patch;savedf[fi]=f[ids]
            sm,h=measure(patch,patchref,temp,localpairs,sel,core,box,len(q)<100)
            sm[16]=np.sum(v*v)/(2*len(q));sm[17]=ep/len(q)
            stats[fi]=sm;bhist[fi//200]+=h;fi+=1
    return coords,savedf,stats,bhist,q,v
