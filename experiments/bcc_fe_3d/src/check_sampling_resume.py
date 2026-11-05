"""Synthetic interruption checks; no MD data, optimizer steps, or scientific results."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import evaluation as ev
from common import ROOT,protocol,write_json,now,sha256


class InjectedInterrupt(BaseException):
    pass


class FakeWhitening:
    def physical(self,z,T):return z


class FakeModel:
    wh=FakeWhitening()


def main():
    temporary=Path(tempfile.mkdtemp(prefix='bcc_fe_sampling_resume_',dir='/tmp'))
    (temporary/'data/processed').mkdir(parents=True)
    (temporary/'data/processed/preconditioner.pt').write_bytes(b'fixture identity only')
    (temporary/'models/force').mkdir(parents=True)
    (temporary/'models/force/step_006000.pt').write_bytes(b'fixture checkpoint identity only')
    control=dict(interrupt=False,batches=0,stress_calls=0,fail_at=None)

    def fake_gaussian(T,count,seed,start_at=0):
        q=np.random.default_rng(seed).normal(size=(count,8,3))
        for start in range(start_at,count,64):
            control['batches']+=1
            if control['interrupt'] and control['batches']==2:raise InjectedInterrupt()
            yield q[start:start+64]

    def fake_neural(model,T,count,seed,**kwargs):
        control['batches']+=1
        if control['interrupt'] and control['batches']==2:raise InjectedInterrupt()
        return np.random.default_rng(seed).normal(size=(count,24381)).astype(np.float32)

    def fake_stress(q,*args):
        control['stress_calls']+=1
        if control['stress_calls']==control['fail_at']:raise ValueError('Injected observable failure')
        mean=q.mean(0)
        return np.outer(mean,mean)+np.eye(3)*q.var(),len(q),None

    def fake_observables(batch,*args,**kwargs):
        for q in batch:yield fake_stress(q)[0],2.0

    rows=[]
    with patch.multiple(ev,ROOT=temporary,protocol=protocol,evaluate_observables=fake_observables,
                        gaussian_samples=fake_gaussian,sample_latent=fake_neural,
                        load_model=lambda *args:(FakeModel(),{})):
        for objective,count,committed in [('gaussian',192,64),('force',1280,512)]:
            step=0 if objective=='gaussian' else 6000
            control.update(interrupt=False,batches=0,stress_calls=0)
            full=ev.generated(objective,step,300,count,purpose='fixture_full',seed=9172)
            reference=np.load(temporary/full['folder']/'stress.npy')
            control.update(interrupt=True,batches=0,stress_calls=0)
            try:ev.generated(objective,step,300,count,purpose='fixture_resume',seed=9172)
            except InjectedInterrupt:pass
            else:raise AssertionError('Interruption did not occur.')
            folder=next((temporary/'results/fixture_resume').glob(f'{objective}_*'))
            progress=json.loads((folder/'progress.json').read_text())
            assert progress['samples']==committed
            prefix=np.load(folder/'stress.npy')[:committed].copy()
            control.update(interrupt=False,batches=0)
            resumed=ev.generated(objective,step,300,count,purpose='fixture_resume',seed=9172)
            actual=np.load(temporary/resumed['folder']/'stress.npy')
            row=dict(method=objective,prefix_preserved=bool(np.array_equal(actual[:committed],prefix)),
                     bitwise_equal_to_uninterrupted=bool(np.array_equal(actual,reference)),
                     stress_evaluations=control['stress_calls'],expected_evaluations=count)
            assert row['prefix_preserved'] and row['bitwise_equal_to_uninterrupted'] and control['stress_calls']==count
            rows.append(row)
        control.update(interrupt=False,batches=0,stress_calls=0,fail_at=70)
        try:ev.generated('gaussian',0,300,192,purpose='fixture_failure',seed=9172)
        except ValueError as error:assert 'Injected observable failure' in str(error)
        else:raise AssertionError('Observable failure did not propagate.')
        folder=next((temporary/'results/fixture_failure').glob('gaussian_*'))
        failure=json.loads((folder/'failure.json').read_text())
        bad=np.load(folder/'failure_configuration.npz')
        expected_bad=np.random.default_rng(9172).normal(size=(192,8,3))[69]
        assert failure['samples_written']==69 and failure['failed_sample_index']==69
        assert int(bad['index'])==69 and np.array_equal(bad['q'],expected_bad)
        assert json.loads((folder/'progress.json').read_text())['samples']==64
        control['fail_at']=None
        # Final 200 K coordinates survive an interruption at the same committed boundary.
        control.update(interrupt=True,batches=0,stress_calls=0)
        try:ev.generated('gaussian',0,200,192,purpose='final',seed=9172)
        except InjectedInterrupt:pass
        else:raise AssertionError('Final-coordinate interruption did not occur.')
        final_folder=next((temporary/'results/final').glob('gaussian_*'))
        prefix=np.load(final_folder/'coordinates.npy')[:64].copy()
        control.update(interrupt=False,batches=0)
        final=ev.generated('gaussian',0,200,192,purpose='final',seed=9172)
        coordinates=np.load(final_folder/'coordinates.npy')
        assert np.array_equal(coordinates[:64],prefix)
        assert np.array_equal(coordinates,np.random.default_rng(9172).normal(size=(192,8,3)))
        assert final['coordinates_file']=='coordinates.npy'
        # A switching model exactly reuses its parent before the switch, without resampling.
        (temporary/'models/force').mkdir(exist_ok=True)
        (temporary/'models/force/step_006000.pt').write_bytes(b'fixture joint identity')
        (temporary/'results/switch_selection.json').write_text(json.dumps(dict(selected_step=6000)))
        control.update(interrupt=False,batches=0,stress_calls=0)
        parent=ev.generated('force',6000,300,1280,purpose='fixture_shared',seed=9172)
        calls=control['stress_calls']
        branch=ev.generated('force_then_denoising',6000,300,1280,purpose='fixture_shared',seed=9172)
        assert parent['folder']==branch['folder'] and branch['shared_prefix_source']=='force'
        assert control['stress_calls']==calls and parent['mean']==branch['mean']
        excluded=False
        try:ev.generated('gaussian',0,200,192,purpose='checkpoint_validation')
        except ValueError:excluded=True
        assert excluded
        # A completed cache must also refuse a different preconditioner identity.
        (temporary/'data/processed/preconditioner.pt').write_bytes(b'changed identity')
        rejected=False
        try:ev.generated('force',6000,300,1280,purpose='fixture_full',seed=9172)
        except RuntimeError as error:rejected='Inputs changed' in str(error)
        assert rejected
    result=dict(at=now(),passed=True,scope='synthetic sampling interruption checks only; no trained models or experimental results',
                cases=rows,changed_input_rejected=rejected,observable_failure_index_verified=True,
                final_200K_coordinates_resumed_exactly=True,shared_prefix_reused_without_resampling=True,
                held_out_selection_rejected=excluded,
                evaluation_sha256=sha256(ROOT/'src/evaluation.py'),
                observable_backend_sha256=sha256(ROOT/'src/observable_parallel.py'),
                fixture_directory=str(temporary))
    write_json(ROOT/'checks/sampling_resume_validation.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
