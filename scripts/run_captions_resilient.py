"""Operational resilience wrapper; the frozen inference code and settings stay identical."""
import json,sqlite3,sys
from pathlib import Path
import run_pope
from atomic_io import atomic_json,event
ROOT=run_pope.ROOT
run_pope.atomic_json=atomic_json
import run_captions as worker
worker.atomic_json=atomic_json
original_status=worker.status
original_export=worker.export

def safe_status(*args,**kwargs):
    try:return original_status(*args,**kwargs)
    except PermissionError as e:
        event('status_update_deferred',error=repr(e),stage=args[0],counts=args[1])
        # The SQLite/raw result was already committed. Retry status on the next result.

def safe_export(*args,**kwargs):
    try:return original_export(*args,**kwargs)
    except PermissionError as e:
        event('export_deferred',error=repr(e),folder=str(args[1]))

worker.status=safe_status;worker.export=safe_export
atomic_json(ROOT/'logs/caption-resilience-layer.json',{'at':run_pope.now(),
    'wrapper_sha256':run_pope.digest(__file__),'atomic_io_sha256':run_pope.digest(ROOT/'scripts/atomic_io.py'),
    'inference_script_sha256':run_pope.digest(ROOT/'scripts/run_captions.py'),
    'changes':['Retry Windows sharing conflicts for atomic metadata replacement',
               'Defer blocked status/exports; preserve already committed SQLite and raw outputs'],
    'inference_parameters_changed':False})
worker.main()
