"""FP64 CUDA EAM/BAOAB backend, accepted only after independent LAMMPS checks.

Every graph block is transactional: a neighbor-skin violation causes an exact
replay from q,v and RNG counter with a fresh list at every step. No frame is
discarded, and no invalid-force trajectory is committed.
"""
from pathlib import Path
import hashlib
import numpy as np
import cupy as cp
from common import KB
from physics import EAM

ACCELERATION=9648.533212331001
# Bind the Python argument interface to the CUDA source at module import.
# Later instances in a long worker must not reread a changed kernel interface.
CUDA_SOURCE=Path(__file__).with_name('eam_cuda.cu').read_text()
BACKEND_SOURCES={'gpu_md.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'eam_cuda.cu':hashlib.sha256(CUDA_SOURCE.encode()).hexdigest()}


class GPUSimulation:
    def __init__(self,q,F,length,temperature,seed,dt=.001,damping=.1,skin=.8,block=16):
        self.cp=cp;self.eam=EAM();self.n=len(q);self.F=np.array(F);self.length=float(length)
        self.source_hashes=BACKEND_SOURCES.copy()
        if not np.allclose(F,[[F[0,0],F[0,1],0],[0,1,0],[0,0,1]]):
            raise ValueError('CUDA backend is restricted to the configured strain family.')
        self.reference=np.array(q);self.q=cp.asarray(q,dtype=cp.float64)
        self.q_saved=cp.empty_like(self.q);self.v_saved=cp.empty_like(self.q)
        self.built=cp.empty_like(self.q);self.force=cp.zeros_like(self.q)
        self.step=cp.zeros(1,dtype=cp.uint64);self.step_saved=cp.zeros_like(self.step)
        self.temperature=temperature;self.seed=np.uint64(seed);self.dt=np.float64(dt)
        self.acc=np.float64(ACCELERATION/self.eam.mass)
        self.decay=np.float64(np.exp(-dt/damping))
        self.noise=np.float64(np.sqrt(KB*temperature*self.acc*(1-self.decay**2)))
        self.v=cp.asarray(np.random.default_rng(seed).normal(
            0,np.sqrt(KB*temperature*self.acc),(self.n,3)))
        self.skin=skin;self.block=block;self.error=cp.zeros(1,dtype=cp.int32)
        self.nc=max(1,int(length*np.linalg.svd(F,compute_uv=False).min()/(self.eam.cutoff+skin)))
        self.cap=128;self.maxnb=256
        self.count=cp.zeros(self.nc**3,dtype=cp.int32)
        self.cellatoms=cp.empty((self.nc**3,self.cap),dtype=cp.int32)
        self.which=cp.empty(self.n,dtype=cp.int32)
        self.numbers=cp.empty(self.n,dtype=cp.int32)
        self.neighbors=cp.empty((self.maxnb,self.n),dtype=cp.int32)
        self.rho=cp.empty(self.n,dtype=cp.float64);self.fp=cp.empty_like(self.rho);self.energy=cp.empty_like(self.rho)
        self.den=cp.asarray(self.eam.density);self.emb=cp.asarray(self.eam.embed);self.rp=cp.asarray(self.eam.rphi)
        module=cp.RawModule(code=CUDA_SOURCE,options=('-std=c++11',))
        self.k={name:module.get_function(name) for name in ['cells','neighbors','density','force','drift','kick']}
        self.grid=((self.n+127)//128,);self.threads=(128,)
        self.geom=(np.int32(self.n),np.int32(self.nc),np.float64(length),np.float64(F[0,0]),np.float64(F[0,1]))
        self.force_geom=(np.float64(length),np.float64(F[0,0]),np.float64(F[0,1]),np.float64(self.eam.cutoff**2))
        self.stream=cp.cuda.Stream(non_blocking=True)
        self.graphs={};self.replays=0
        self.rebuild();self.evaluate();cp.cuda.get_current_stream().synchronize()
        # Warm kernels before capture; save and restore the complete stochastic state.
        self.q_saved[:]=self.q;self.v_saved[:]=self.v;self.step_saved[:]=self.step
        self.one_step();cp.cuda.get_current_stream().synchronize()
        self.q[:]=self.q_saved;self.v[:]=self.v_saved;self.step[:]=self.step_saved
        self.rebuild();self.evaluate();cp.cuda.get_current_stream().synchronize()
        with self.stream:
            for count in [1,block]:
                self.stream.begin_capture()
                for _ in range(count): self.one_step()
                self.graphs[count]=self.stream.end_capture()

    def rebuild(self):
        self.count.fill(0);self.error.fill(0)
        self.k['cells'](self.grid,self.threads,(self.q,*self.geom,np.int32(self.cap),self.count,self.cellatoms,self.which,self.error))
        self.k['neighbors'](self.grid,self.threads,(self.q,*self.geom,np.float64((self.eam.cutoff+self.skin)**2),
            np.int32(self.cap),self.count,self.cellatoms,self.which,np.int32(self.maxnb),self.numbers,self.neighbors,self.error))
        self.built[:]=self.q
        if int(self.error.get()[0]): raise RuntimeError('CUDA cell/neighbor capacity exceeded; no trajectory committed.')

    def evaluate(self):
        e=self.eam
        self.k['density'](self.grid,self.threads,(self.q,np.int32(self.n),self.numbers,self.neighbors,*self.force_geom,
            self.den,self.emb,np.int32(len(e.density)),np.int32(len(e.embed)),np.float64(e.dr),np.float64(e.drho),
            self.rho,self.fp,self.energy,self.error))
        self.k['force'](self.grid,self.threads,(self.q,np.int32(self.n),self.numbers,self.neighbors,*self.force_geom,
            self.den,self.rp,np.int32(len(e.density)),np.float64(e.dr),self.fp,self.force,self.energy))

    def one_step(self):
        self.k['drift'](self.grid,self.threads,(self.q,self.v,self.force,self.built,np.int32(self.n),
            self.dt,self.acc,self.decay,self.noise,np.float64((self.skin/2)**2),self.seed,self.step,self.error))
        self.evaluate()
        self.k['kick'](((3*self.n+127)//128,),self.threads,(self.v,self.force,np.int32(self.n),self.dt,self.acc,self.step))

    def run(self,steps):
        while steps:
            count=self.block if steps>=self.block else 1
            self.rebuild();self.evaluate()
            self.q_saved[:]=self.q;self.v_saved[:]=self.v;self.step_saved[:]=self.step
            cp.cuda.get_current_stream().synchronize()
            self.graphs[count].launch(self.stream);self.stream.synchronize()
            flag=int(self.error.get()[0])
            if flag & 2:
                self.replays+=1
                self.q[:]=self.q_saved;self.v[:]=self.v_saved;self.step[:]=self.step_saved
                for _ in range(count):
                    self.rebuild();self.evaluate();self.one_step()
                    flag=int(self.error.get()[0])
                    if flag: raise RuntimeError(f'Neighbor safety failed even at one-step rebuild: {flag}')
            elif flag & (1|4):
                raise RuntimeError(f'CUDA EAM domain/capacity failure {flag}')
            steps-=count

    def positions(self,aligned=True):
        q=self.q.get()
        if aligned:q-=np.mean(q-self.reference,axis=0)
        return q

    def forces(self):
        return self.force.get()

    def observables(self):
        return dict(temperature_K=float(cp.mean(self.v*self.v).get()/(KB*self.acc)),
                    energy_eV_per_atom=float(cp.mean(self.energy).get()),replayed_blocks=self.replays)
