"""Check the manuscript temperature split without training or running MD."""
from types import SimpleNamespace
from unittest.mock import patch
import math
import numpy as np
import torch
import evaluation
from learning import Preconditioner
from common import ROOT, protocol, now, sha256, write_json


def main():
    p=protocol();temps=p['temperatures_K']['train']
    assert temps==p['temperatures_K']['validation']==[50,150,250,350]
    assert p['temperatures_K']['held_out']==[100,200,300,400]
    assert p['temperatures_K']['final_evaluation']==[50,100,150,200,250,300,350,400]
    assert p['generation']['selection_temperatures_K']==temps
    assert p['models']['warm_start']['selection_temperatures_K']==temps
    dummy=SimpleNamespace(temperatures=torch.tensor(temps,dtype=torch.float32),
        means=torch.tensor(temps,dtype=torch.float32)[:,None]/100)
    prediction=Preconditioner.mean(dummy,torch.tensor([100.,200.,300.,400.]))
    torch.testing.assert_close(prediction[:,0],torch.tensor([1.,2.,3.,3.5]))
    used=[]
    def concatenate(paths,name):
        T=paths[0];used.append(T)
        return np.stack([np.full(24381,T/100+.02,dtype=np.float32),
                         np.full(24381,T/100-.02,dtype=np.float32)])
    def noise(*shape,device=None,generator=None,**kwargs):
        out=torch.zeros(*shape,device=device)
        if shape[1]==2:out[:,0]=1
        return out
    with patch.object(evaluation,'paths_for',side_effect=lambda role,T:[T]), \
         patch.object(evaluation,'concatenate',side_effect=concatenate), \
         patch.object(evaluation.torch,'randn',side_effect=noise):
        extrap=np.concatenate(list(evaluation.gaussian_samples(400,4,17)))
        assert used==[350],used
        expected=3.5+.02*math.sqrt(400/350/2)
        np.testing.assert_allclose(extrap,expected,rtol=0,atol=1e-6)
        used.clear()
        interp=np.concatenate(list(evaluation.gaussian_samples(200,4,17)))
        assert used==[150,250],used
        expected=2+.02*(math.sqrt(.5*200/150/2)+math.sqrt(.5*200/250/2))
        np.testing.assert_allclose(interp,expected,rtol=0,atol=1e-6)
    write_json(ROOT/'checks/temperature_scope_validation.json',dict(passed=True,at=now(),
        protocol_sha256=sha256(ROOT/'src/settings.py'),evaluation_sha256=sha256(ROOT/'src/evaluation.py'),
        training_temperatures=temps,test_temperatures=p['temperatures_K']['held_out'],
        extrapolation_temperature_K=400,preconditioner_endpoint_mean_verified=True,
        gaussian_endpoint_mean_and_thermal_covariance_scaling_verified=True,
        scope='Synthetic algebra fixture, no scientific training or MD. Checks training-only inputs for Gaussian extrapolation.'))
    print('Four-temperature scope and Gaussian extrapolation checks passed.',flush=True)


if __name__=='__main__':main()
