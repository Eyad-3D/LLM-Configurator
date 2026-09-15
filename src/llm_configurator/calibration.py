"""Versioned calibration cache and hard-bounded worker lifecycle."""
from datetime import datetime, timezone
import json
import math
import os
import subprocess
import sys

from .domain import now

VERSION = 1
MAX_AGE_DAYS = 7


def valid(record, hardware):
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(record['timestamp'])).total_seconds()
        return (record['version'] == VERSION and record['fingerprint'] == hardware['fingerprint']
                and 0 <= age < MAX_AGE_DAYS * 86400)
    except (KeyError, TypeError, ValueError):
        return False


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def calibrate(hardware, timeout=12):
    env = os.environ.copy()
    for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        env[key] = '1'
    command = [sys.executable, '-m', 'llm_configurator.calibration_worker']
    timed_out = False
    try:
        completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=timeout,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        output = completed.stdout
    except subprocess.TimeoutExpired as error:
        timed_out = True
        output = error.stdout or ''
        if isinstance(output, bytes):
            output = output.decode('utf-8', errors='replace')
    record = {'cpu': None, 'gpus': {}, 'warnings': []}
    for line in output.splitlines():
        try:
            candidate = json.loads(line)
            if isinstance(candidate, dict) and 'cpu' in candidate and isinstance(candidate.get('gpus'), dict):
                record = candidate
        except ValueError:
            continue
    if timed_out:
        record['warnings'].append('Calibration time limit reached; completed measurements retained.')
    if not record['cpu'] and not record['gpus']:
        record['warnings'].append('No usable calibration. Install dependencies and try Recalibrate.')
    record.update(version=VERSION, timestamp=now(), fingerprint=hardware['fingerprint'])
    return record
