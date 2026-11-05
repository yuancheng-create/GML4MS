"""Required quantitative EAM/IK checks; no model training."""
import json
import time
import numpy as np
from scipy.optimize import minimize_scalar
from common import ROOT, protocol, write_json, now, POTENTIAL, sha256
from physics import EAM, bcc, stress, buffer_margin, bond_weights
from backend import Simulation


def main():
    p = protocol()
    e = EAM()
    def energy_per_atom(a):
        q, _ = bcc(6,a,np.eye(3),vacancy=False)
        return e.evaluate(q,np.eye(3),6*a)['energy']/len(q)
    fit=minimize_scalar(energy_per_atom,bounds=(2.82,2.90),method='bounded',
                        options={'xatol':1e-10})
    a=float(fit.x)
    F=np.array(p['geometry']['F'])
    rng=np.random.default_rng(815832)
    q,_=bcc(6,a,F)
    q+=rng.normal(0,.015,q.shape)
    sim=Simulation(q,F,6*a,threads=4,logfile=ROOT/'logs/physics_validation.log')
    v=e.evaluate(q,F,6*a)
    sigma,_,_=stress(q,v,F,a,0,whole_periodic=True)
    result=dict(date=now(),potential_sha256=sha256(POTENTIAL),
        lattice_constant_header_A=e.a0,lattice_constant_relaxed_A=a,
        cohesive_energy_eV_per_atom=float(fit.fun),
        energy_difference_eV=abs(v['energy']-sim.lmp.get_thermo('pe')),
        force_max_difference_eV_per_A=float(abs(v['forces']-sim.forces()).max()),
        stress_max_difference_GPa=float(abs(sigma-sim.potential_stress()).max()),
        net_force_eV_per_A=np.sum(v['forces'],axis=0).tolist())
    fd=[]
    for atom,axis in [(0,0),(123,1),(207,2)]:
        h=1e-5
        plus=q.copy();minus=q.copy();plus[atom,axis]+=h;minus[atom,axis]-=h
        f=-(e.evaluate(plus,F,6*a)['energy']-e.evaluate(minus,F,6*a)['energy'])/(2*h)
        fd.append(abs(f-v['forces'][atom,axis]))
    result['finite_difference_force_max_error_eV_per_A']=float(max(fd))
    sim.close()
    # Exact elementary line-intersection cases, including both endpoints outside C.
    weights=bond_weights(np.array([[-2.,0,0],[-2.,2,0],[0.,0,0]]),
                         np.array([[4.,0,0],[4.,0,0],[.5,0,0]]),1.)
    assert np.allclose(weights,[.5,0,1])
    q,sites=bcc(40,a,F)
    q+=rng.normal(0,.04,q.shape)
    ids=np.flatnonzero(np.all((sites>=-8)&(sites<8),axis=1))
    large=e.evaluate(q,F,40*a)
    region=e.evaluate(q[ids])
    sl,cl,wl=stress(q,large,F,a,300)
    sr,cr,wr=stress(q[ids],region,F,a,300)
    result.update(retained_atoms=len(ids),large_atoms=len(q),
        core_count_large=cl,core_count_retained=cr,
        full_vs_retained_stress_max_error_GPa=float(abs(sl-sr).max()),
        buffer_slack_lower_bound_A=buffer_margin(q,ids,F,a,e.cutoff),
        reference_stress_GPa=sl.tolist())
    result['passed']=(result['energy_difference_eV']<1e-7 and
        result['force_max_difference_eV_per_A']<1e-8 and
        result['finite_difference_force_max_error_eV_per_A']<1e-5 and
        result['stress_max_difference_GPa']<1e-5 and
        result['full_vs_retained_stress_max_error_GPa']<1e-5 and
        result['buffer_slack_lower_bound_A']>0 and cl==cr)
    write_json(ROOT/'checks/physics_validation.json',result)
    print(json.dumps(result,indent=2),flush=True)
    if not result['passed']:
        raise RuntimeError('Physical validation failed; do not start production.')
    if abs(a-p['potential']['a0_verified_A']) > 1e-6:
        raise RuntimeError('Lattice constant differs from the release protocol; review before MD.')


if __name__ == '__main__':
    main()
