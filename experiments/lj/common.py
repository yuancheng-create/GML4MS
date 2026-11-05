"""Paths, manuscript settings and atomic run records for the LJ experiment."""
from pathlib import Path
import hashlib
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
SAM = ROOT.parents[1]
TRAIN_TEMPS = [.02, .04, .06, .08, .10]
TEMPS = [i / 100 for i in range(2, 11)]
A = 1.1161533029896453

def resolve_recorded_path(path):
    path = Path(path)
    return path if path.is_absolute() else SAM / path


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def digest(path):
    h = hashlib.sha256()
    with resolve_recorded_path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def case_root(vacancies):
    return ROOT / 'cases' / f'vacancy_{vacancies}'


PROTOCOL = {
    'vacancy_counts': [7],
    'reference_n': 48,
    'retained_n': 18,
    'observation_n': 8,
    'training_temperatures': TRAIN_TEMPS,
    'test_temperatures': TEMPS,
    'training_repetitions': 1,
    'objectives': ['force', 'noise', 'joint'],
    'updates_per_model': 100000,
    'batch_size': 256,
    'joint_weights': {'force': .5, 'noise': 4.0},
    'checkpoint_rule': 'Final EMA at 100000 updates; 6000-update comparison saved separately',
    'generation_samples_per_seed_temperature': 50000,
    'generation_steps': 25,
    'generation_precision': 'BF16 network, FP32 state, FP64 coordinates and observables',
    'md_equilibration_steps': 100000,
    'md_initial_production_steps': 1000000,
    'md_extension_production_steps': 1000000,
    'md_save_stride': 100,
    'md_dt': .002,
    'md_gamma': 1.0,
    'no_filtering_generated_samples': True,
}
