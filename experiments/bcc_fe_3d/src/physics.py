"""Independent single-element EAM and spatial IK evaluator in physical units.

Hermite interpolation uses the finite-difference knot slopes specified by
LAMMPS pair_eam.cpp, so the tabulated potential has the same interpretation.
The implementation below is an independent polynomial/chain-rule evaluation.
"""
import numpy as np
from scipy.spatial import cKDTree
from numba import njit
from common import KB, EV_A3_GPA, POTENTIAL


def hermite_coefficients(y):
    m = np.empty_like(y)
    m[0], m[-1] = y[1]-y[0], y[-1]-y[-2]
    m[1], m[-2] = (y[2]-y[0])/2, (y[-1]-y[-3])/2
    m[2:-2] = (y[:-4]-y[4:]+8*(y[3:-1]-y[1:-3]))/12
    dy = np.diff(y)
    return np.ascontiguousarray(np.stack((y[:-1], m[:-1],
        3*dy-2*m[:-1]-m[1:], m[:-1]+m[1:]-2*dy), axis=1))


class EAM:
    def __init__(self, path=POTENTIAL):
        lines = path.read_text().splitlines()
        header = lines[4].split()
        self.nrho, self.drho, self.nr, self.dr, self.cutoff = (
            int(header[0]), float(header[1]), int(header[2]), float(header[3]), float(header[4]))
        self.mass, self.a0 = map(float, lines[5].split()[1:3])
        v = np.fromstring(' '.join(lines[6:]), sep=' ')
        assert len(v) == self.nrho+2*self.nr
        self.embed = hermite_coefficients(v[:self.nrho])
        self.density = hermite_coefficients(v[self.nrho:self.nrho+self.nr])
        self.rphi = hermite_coefficients(v[self.nrho+self.nr:])

    def evaluate(self, q, F=None, length=None):
        pairs, dr = pair_list(q, self.cutoff, F, length)
        energy, force, virial, rho, fpair, distances = evaluate_pairs(
            len(q), pairs, dr, self.embed, self.density, self.rphi,
            self.drho, self.dr, self.cutoff)
        if rho.max(initial=0) > (self.nrho-1)*self.drho:
            raise ValueError('EAM density outside supplied table; do not silently filter samples.')
        return dict(energy=float(energy), forces=force, virial=virial, density=rho,
                    pairs=pairs, dr=dr, pair_forces=fpair, distances=distances)


def bcc(n, a, F, vacancy=True):
    k = np.stack(np.meshgrid(*(np.arange(-n//2, n//2),)*3,
                            indexing='ij'), axis=-1).reshape(-1, 3)
    # Cell-major order with two basis atoms; this order is also the network map.
    sites = (k[:, None, :] + np.array([[0., 0., 0.], [.5, .5, .5]]))[...].reshape(-1, 3)
    keep = np.ones(len(sites), dtype=bool)
    if vacancy:
        keep = np.sum(sites*sites, axis=1) > 4+1e-12
        assert (~keep).sum() == 65
    sites = sites[keep]
    return np.ascontiguousarray(sites*a@np.asarray(F).T), sites


def pair_list(q, cutoff, F=None, length=None):
    q = np.asarray(q, dtype=np.float64)
    if length is None:
        pairs = cKDTree(q).query_pairs(cutoff, output_type='ndarray')
        dr = q[pairs[:, 0]]-q[pairs[:, 1]]
    else:
        F = np.asarray(F)
        x = q@np.linalg.inv(F).T
        radius = cutoff/np.linalg.svd(F, compute_uv=False).min()
        if radius >= length/2:
            raise ValueError('Periodic cell too small for this minimum-image evaluator.')
        wrapped = np.mod(np.mod(x, length), length)
        pairs = cKDTree(wrapped, boxsize=length).query_pairs(radius, output_type='ndarray')
        dx = x[pairs[:, 0]]-x[pairs[:, 1]]
        dx -= length*np.rint(dx/length)
        dr = dx@F.T
        keep = np.sum(dr*dr, axis=1) < cutoff*cutoff
        pairs, dr = pairs[keep], dr[keep]
    return np.ascontiguousarray(pairs, dtype=np.int64), np.ascontiguousarray(dr)


@njit(cache=True)
def spline(x, step, c):
    t = x/step
    j = min(max(int(t), 0), len(c)-1)
    u = min(max(t-j, 0.), 1.)
    a, b, d, e = c[j]
    value = a+u*(b+u*(d+u*e))
    deriv = (b+u*(2*d+3*e*u))/step
    if t > len(c):
        value += (x-step*len(c))*deriv
    return value, deriv


@njit(cache=True)
def evaluate_pairs(n, pairs, dr, emb, den, rp, drho, step, cutoff):
    rho = np.zeros(n)
    distance = np.sqrt(np.sum(dr*dr, axis=1))
    for k in range(len(pairs)):
        i, j = pairs[k]
        w, _ = spline(distance[k], step, den)
        rho[i] += w
        rho[j] += w
    energy = 0.
    fp = np.empty(n)
    for i in range(n):
        v, fp[i] = spline(rho[i], drho, emb)
        energy += v
    forces = np.zeros((n, 3))
    pairforces = np.zeros_like(dr)
    virial = np.zeros((3, 3))
    for k in range(len(pairs)):
        i, j = pairs[k]
        r = distance[k]
        z, dz = spline(r, step, rp)
        _, dw = spline(r, step, den)
        phi = z/r
        energy += phi
        f = -((fp[i]+fp[j])*dw + (dz-phi)/r)/r
        for d in range(3):
            fd = f*dr[k, d]
            pairforces[k, d] = fd
            forces[i, d] += fd
            forces[j, d] -= fd
            for e in range(3):
                virial[d, e] += fd*dr[k, e]
    return energy, forces, virial, rho, pairforces, distance


@njit(cache=True)
def bond_weights(xj, dx, half):
    weights = np.zeros(len(xj))
    for k in range(len(xj)):
        lo, hi = 0., 1.
        for d in range(3):
            if abs(dx[k, d]) < 1e-14:
                if xj[k, d] < -half or xj[k, d] >= half:
                    hi = -1.
                    break
            else:
                a = (-half-xj[k, d])/dx[k, d]
                b = (half-xj[k, d])/dx[k, d]
                lo = max(lo, min(a, b))
                hi = min(hi, max(a, b))
        weights[k] = max(0., hi-lo)
    return weights


def stress(q, evaluation, F, a, temperature, core_n=6, whole_periodic=False):
    inverse = np.linalg.inv(F)
    x = q@inverse.T
    half = core_n*a/2
    volume = np.linalg.det(F)*(core_n*a)**3
    if whole_periodic:
        weights = np.ones(len(evaluation['pairs']))
        count = len(q)
    else:
        count = int(np.all((x >= -half) & (x < half), axis=1).sum())
        pairs = evaluation['pairs']
        weights = bond_weights(x[pairs[:, 1]], evaluation['dr']@inverse.T, half)
    virial = np.einsum('ni,nj,n->ij', evaluation['pair_forces'], evaluation['dr'], weights)
    sigma = -(virial+KB*temperature*count*np.eye(3))/volume*EV_A3_GPA
    return sigma, count, weights


def buffer_margin(q, retained, F, a, cutoff, core_n=6):
    """Conservative lower bound for distance of discarded atoms to current C."""
    outside = np.ones(len(q), dtype=bool)
    outside[retained] = False
    x = q[outside]@np.linalg.inv(F).T
    distance = np.linalg.norm(np.maximum(abs(x)-core_n*a/2, 0), axis=1)
    return float(distance.min()*np.linalg.svd(F, compute_uv=False).min()-2*cutoff)
