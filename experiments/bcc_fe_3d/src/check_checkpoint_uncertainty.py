"""Analytic correlated-noise checks; no new MD, training or generated samples."""
import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import evaluation as ev
from common import ROOT,now,sha256,write_json


def main():
    temporary=Path(tempfile.mkdtemp(prefix='bcc_fe_score_uncertainty_',dir='/tmp'))
    count=257;noise=np.random.default_rng(88341).normal(size=count)*.2;noise-=noise.mean()
    bias=np.array([[1.,.2,0],[.2,.5,.1],[0,.1,-.3]])
    direction=np.array([[.7,-.3,.2],[-.3,.4,0],[.2,0,.1]])
    norm=float(np.sqrt((bias[0,0]**2+bias[0,1]**2)/2))
    projection=float((bias[0,0]*direction[0,0]+bias[0,1]*direction[0,1])/(2*norm))
    md_se=.003;cov_md=md_se**2*np.outer(direction.ravel(),direction.ravel());cases=[]
    with patch.object(ev,'ROOT',temporary):
        for name,coefficients in [('positive',[1,1,1]),('partial_cancellation',[1,-1,1]),('full_cancellation',[1,-1,0])]:
            reports=[];references=[]
            for index,coefficient in enumerate(coefficients):
                folder=temporary/name/str(index);folder.mkdir(parents=True)
                values=bias[None]+coefficient*noise[:,None,None]*direction[None]
                np.save(folder/'stress.npy',values)
                reports.append(dict(samples=count,seed=811872,mean=values.mean(0).tolist(),
                    covariance_of_mean=(np.cov(values.reshape(count,9),rowvar=False)/count).tolist(),
                    folder=str(folder.relative_to(temporary))))
                references.append(dict(mean=np.zeros((3,3)).tolist(),covariance_of_mean=cov_md.tolist()))
            score,se=ev.selection_score(reports,references)
            expected=abs(projection)*np.sqrt(noise.var(ddof=1)/count*(sum(coefficients)/3)**2+md_se**2/3)
            naive=abs(projection)*np.sqrt(noise.var(ddof=1)/count*sum(c*c for c in coefficients)/9+md_se**2/3)
            assert np.isclose(score,norm,rtol=1e-12,atol=1e-14)
            assert np.isclose(se,expected,rtol=1e-11,atol=1e-14),(name,se,expected)
            cases.append(dict(case=name,score=score,standard_error=se,analytic_standard_error=float(expected),
                              incorrect_independent_temperature_standard_error=float(naive),passed=True))
        rejected={}
        for issue in ['seed','count','summary','zero_norm']:
            altered=copy.deepcopy(reports);ref=copy.deepcopy(references)
            if issue=='seed':altered[0]['seed']+=1
            elif issue=='count':altered[0]['samples']-=1
            elif issue=='summary':altered[0]['mean'][0][0]+=.01
            else:
                for source,target in zip(altered,ref):target['mean']=copy.deepcopy(source['mean'])
            try:ev.selection_score(altered,ref)
            except (ValueError,RuntimeError) as error:rejected[issue]=str(error)
            else:raise AssertionError(f'Invalid case was not rejected: {issue}')
    result=dict(at=now(),passed=True,scope='Synthetic analytic uncertainty and data-consistency checks only; no new physical experiments.',
        evaluation_sha256=sha256(ROOT/'src/evaluation.py'),cases=cases,rejected_cases=rejected,
        interpretation='Retains cross-temperature generated covariance from shared latent noise and adds independent validation-MD covariance; point score and prescribed one-standard-error selection rule are unchanged.',
        limitation='Delta-method approximation at nonzero stress bias; this is a selection heuristic, not a confidence proof of a unique optimum.',
        fixture_directory=str(temporary))
    write_json(ROOT/'checks/checkpoint_uncertainty_validation.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
