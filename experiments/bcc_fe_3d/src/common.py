"""Paths, atomic metadata and physical constants for the BCC Fe experiment."""
from pathlib import Path
import json
from copy import deepcopy
from settings import SETTINGS
import os
import hashlib
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
KB = 8.617333262145e-5  # eV/K
EV_A3_GPA = 160.2176634
POTENTIAL = ROOT / 'potentials/Fe_Mishin2006.eam.alloy'


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.pending')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def protocol():
    p = deepcopy(SETTINGS)
    for name in ['checks', 'logs', 'data', 'models', 'results', 'provenance', 'potentials']:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    return p


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()
