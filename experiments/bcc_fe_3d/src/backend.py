"""LAMMPS reference backend. MPI is initialized by the supplied wheel."""
import ctypes
import os
import sys
import numpy as np
from common import ROOT, POTENTIAL

# Load the private MPI library without modifying the user's shell environment.
from pathlib import Path
_mpi = Path(os.environ.get('SAM_MPI_LIBRARY', str(Path(sys.prefix)/'lib/libmpi.so.12')))
if _mpi.exists():
    ctypes.CDLL(str(_mpi), mode=ctypes.RTLD_GLOBAL)
from lammps import lammps


class Simulation:
    def __init__(self, q, F, length, threads=1, logfile=None, accelerator='auto'):
        self.F, self.length = np.array(F), float(length)
        self.reference = np.array(q, dtype=np.float64)
        self.n = len(q)
        args = ['-screen', 'none', '-log', str(logfile) if logfile else 'none']
        self.lmp = lammps(cmdargs=args)
        l = self.lmp
        styles = l.available_styles('pair')
        if accelerator == 'auto':
            self.style = 'eam/alloy/omp' if 'eam/alloy/omp' in styles and threads > 1 else 'eam/alloy'
        else:
            self.style = accelerator
        l.commands_list(['units metal', 'atom_style atomic', 'boundary p p p',
                         'atom_modify map array sort 0 0.0'])
        if self.style.endswith('/omp'):
            l.command(f'package omp {threads}')
        H = self.F*length
        assert np.allclose(H, np.triu(H))
        origin = -H.sum(axis=1)/2
        diag = np.diag(H)
        l.command('region simulation prism '+ ' '.join(map(str,[
            origin[0], origin[0]+diag[0], origin[1], origin[1]+diag[1],
            origin[2], origin[2]+diag[2], H[0,1], H[0,2], H[1,2]]))+' units box')
        l.command('create_box 1 simulation')
        made = l.create_atoms(self.n, np.arange(1,self.n+1,dtype=np.int32),
                             np.ones(self.n,dtype=np.int32), np.asarray(q).ravel())
        if made != self.n:
            raise RuntimeError(f'Created {made} atoms instead of {self.n}')
        l.commands_list([f'pair_style {self.style}', f'pair_coeff * * {POTENTIAL} Fe',
                         'neighbor 0.6 bin', 'neigh_modify every 1 delay 0 check yes',
                         'compute configurational all pressure NULL virial',
                         'thermo_style custom step pe temp pxx pyy pzz pxy pxz pyz',
                         'thermo 1000', 'thermo_modify lost error norm no', 'run 0'])

    def positions(self, aligned=True):
        q = np.ctypeslib.as_array(self.lmp.gather_atoms('x',1,3)).copy().reshape(-1,3)
        # Solid-basin unwrapping by fixed atom label; large jumps are audited separately.
        du = (q-self.reference)@np.linalg.inv(self.F).T
        du -= self.length*np.rint(du/self.length)
        q = self.reference+du@self.F.T
        if aligned:
            q -= (q-self.reference).mean(axis=0)
        return q

    def forces(self):
        return np.ctypeslib.as_array(self.lmp.gather_atoms('f',1,3)).copy().reshape(-1,3)

    def potential_stress(self):
        p = self.lmp.extract_compute('configurational', 0, 1)
        return -1e-4*np.array([[p[0],p[3],p[4]],[p[3],p[1],p[5]],[p[4],p[5],p[2]]])

    def set_positions(self, q):
        array = np.ascontiguousarray(q, dtype=np.float64)
        self.lmp.scatter_atoms('x', 1, 3, array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        self.lmp.command('run 0')

    def relax(self, maxiter=3000):
        self.lmp.commands_list(['min_style fire','min_modify dmax 0.1',
            f'minimize 0 1e-8 {maxiter} {maxiter*10}', 'run 0'])
        return self.positions()

    def close(self):
        self.lmp.close()
