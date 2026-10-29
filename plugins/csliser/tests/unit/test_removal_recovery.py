import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from csliser.application.removal_recovery import RecoverRemovals, discover_removal_receipts, recover_removal

ROOT = Path(__file__).resolve().parents[4]
CHILD = r'''
import os,sys
from pathlib import Path
from csliser.application import file_transfer as transfer
from csliser.application.use_cases import ExecuteTransferPlan
from csliser.domain.models import FileOperation,ProcessingConfig,SelectionMode,SourceFolder
from csliser.domain.planner import build_operation_plan
folder=Path(sys.argv[1]); phase=sys.argv[2]
operation=FileOperation(sys.argv[3] if len(sys.argv)>3 else 'delete')
plan=build_operation_plan(ProcessingConfig(sources=(SourceFolder(folder,('.jpg',)),),frame_expression='1',selection_mode=SelectionMode.FULL_RANGE,operation=operation,destination=folder.parent/'copies'))
rename=transfer._rename_new
def crash_rename(source,destination):
 if source!=plan.operations[0].source:return rename(source,destination)
 if operation==FileOperation.MOVE and destination==plan.operations[0].destination:
  import errno
  raise OSError(errno.EXDEV,'controlled cross-volume move')
 if phase=='before':os._exit(73)
 rename(source,destination)
 if phase=='live':
  print('quarantined',flush=True)
  sys.stdin.buffer.read(1)
 if phase in ('after','live'):os._exit(73)
transfer._rename_new=crash_rename
if phase=='removed':
 unlink=Path.unlink
 def crash_unlink(path,*args,**kwargs):
  result=unlink(path,*args,**kwargs)
  if '.csliser-remove-' in path.name and not path.name.endswith('.json'):os._exit(73)
  return result
 Path.unlink=crash_unlink
result=ExecuteTransferPlan().execute(plan)
raise SystemExit(0 if result.ok else 2)
'''


def environment():
    return dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT/'src'), str(ROOT/'plugins/csliser/src')]))


def interrupted(tmp_path, phase='after', operation='delete', filename='frame_1.jpg'):
    folder = tmp_path / 'source'
    folder.mkdir()
    source = folder / 'frame_1.jpg'
    payload = b'original microscope bytes' * 10000
    source.write_bytes(payload)
    if filename != source.name:
        source = source.rename(folder / filename)
    result = subprocess.run([sys.executable, '-c', CHILD, str(folder), phase, operation], env=environment(),
                            capture_output=True, timeout=20)
    assert result.returncode == 73, result.stderr
    paths = discover_removal_receipts(folder).operations
    assert len(paths) == 1
    return source, paths[0], payload


def test_actual_process_exit_after_quarantine_can_restore_exact_source(tmp_path):
    source, receipt, payload = interrupted(tmp_path)
    assert not source.exists() and receipt.with_name(receipt.name[:-5]).read_bytes() == payload
    assert recover_removal(receipt) == 'restored'
    assert source.read_bytes() == payload and not receipt.exists()
    assert RecoverRemovals().execute(discover_removal_receipts(source.parent)).ok


@pytest.mark.parametrize('phase,status,exists', [('before', 'source_unchanged', True), ('removed', 'already_removed', False)])
def test_actual_exit_on_either_side_of_removal_is_reconciled(tmp_path, phase, status, exists):
    source, receipt, payload = interrupted(tmp_path, phase)
    assert recover_removal(receipt) == status
    assert source.exists() == exists and not receipt.exists()
    if exists:
        assert source.read_bytes() == payload


def test_live_process_receipt_cannot_be_recovered(tmp_path):
    folder = tmp_path/'source'
    folder.mkdir()
    source = folder/'frame_1.jpg'
    source.write_bytes(b'live operation')
    child = subprocess.Popen([sys.executable, '-c', CHILD, str(folder), 'live'], env=environment(),
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        # Child prints only after durable receipt and real native rename.
        assert child.stdout.readline().strip() == b'quarantined'
        receipt, = discover_removal_receipts(folder).operations
        with pytest.raises(OSError):
            recover_removal(receipt)
        assert not source.exists() and receipt.with_name(receipt.name[:-5]).exists()
        child.communicate(input=b'x', timeout=15)
        assert child.returncode == 73
        assert recover_removal(receipt) == 'restored'
        assert source.read_bytes() == b'live operation'
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=10)


def test_recovery_never_overwrites_a_new_source(tmp_path):
    source, receipt, payload = interrupted(tmp_path)
    source.write_bytes(b'new independent source')
    result = RecoverRemovals().execute(discover_removal_receipts(source.parent))
    assert result.errors and result.completed == 0
    assert source.read_bytes() == b'new independent source'
    assert receipt.with_name(receipt.name[:-5]).read_bytes() == payload and receipt.exists()


def test_same_size_same_metadata_tampering_is_rejected_by_sha(tmp_path):
    source, receipt, payload = interrupted(tmp_path)
    quarantine = receipt.with_name(receipt.name[:-5])
    value = quarantine.stat()
    quarantine.write_bytes(b'x' * len(payload))
    os.utime(quarantine, ns=(value.st_atime_ns, value.st_mtime_ns))
    with pytest.raises(OSError, match='SHA-256'):
        recover_removal(receipt)
    assert not source.exists() and quarantine.read_bytes() == b'x' * len(payload) and receipt.exists()


@pytest.mark.parametrize('corruption', ['truncated', 'escape', 'wrong-version', 'oversized'])
def test_invalid_receipt_does_not_move_or_delete_payloads(tmp_path, corruption):
    source, receipt, payload = interrupted(tmp_path)
    data = json.loads(receipt.read_text())
    if corruption == 'escape':
        data['source'] = '../outside.jpg'
    elif corruption == 'wrong-version':
        data['version'] = True
    receipt.write_text('{' if corruption == 'truncated' else ('x' * 65537 if corruption == 'oversized' else json.dumps(data)))
    result = RecoverRemovals().execute(discover_removal_receipts(source.parent))
    assert result.errors and result.completed == 0
    assert not source.exists() and receipt.with_name(receipt.name[:-5]).read_bytes() == payload


def test_cancelled_recovery_preserves_quarantine_and_receipt(tmp_path):
    source, receipt, payload = interrupted(tmp_path)
    result = RecoverRemovals().execute(discover_removal_receipts(source.parent), cancelled=lambda: True)
    assert result.cancelled and result.completed == 0 and receipt.exists()
    assert not source.exists() and hashlib.sha256(receipt.with_name(receipt.name[:-5]).read_bytes()).digest() == hashlib.sha256(payload).digest()


def test_cross_volume_move_interruption_restores_source_and_preserves_verified_copy(tmp_path):
    source, receipt, payload = interrupted(tmp_path, operation='move')
    destination = tmp_path/'copies/jpg_source/frame_1.jpg'
    assert destination.read_bytes() == payload
    assert recover_removal(receipt) == 'restored'
    assert source.read_bytes() == destination.read_bytes() == payload


@pytest.mark.skipif(os.name == 'nt', reason='Colon is a legal POSIX filename, forbidden on Windows')
def test_unsupported_posix_colon_is_refused_before_source_mutation(tmp_path):
    folder = tmp_path/'source'
    folder.mkdir()
    source = folder/'chip:frame_1.jpg'
    source.write_bytes(b'preserve unsupported filename')
    result = subprocess.run([sys.executable, '-c', CHILD, str(folder), 'after', 'delete'],
                            env=environment(), capture_output=True, timeout=20)
    # Kraken's existing cross-platform safe_files contract rejects colon even
    # on POSIX. A new receipt must not turn that refusal into data mutation.
    assert result.returncode == 2, result.stderr
    assert source.read_bytes() == b'preserve unsupported filename'
    assert not discover_removal_receipts(folder).operations


def test_actual_cli_recovers_without_loading_gui(tmp_path):
    source, receipt, payload = interrupted(tmp_path)
    result = subprocess.run([sys.executable, str(ROOT/'plugins/csliser/__main__.py'),
                             '--recover-source', str(source.parent)], env=environment(), capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['ok'] and report['completed'] == 1 and report['outcomes'][0]['status'] == 'restored'
    assert source.read_bytes() == payload and not receipt.exists()


def test_failure_to_flush_receipt_prevents_source_mutation(tmp_path, monkeypatch):
    from csliser.application.file_transfer import remove_verified_source
    from csliser.domain.models import FileOperation, ProcessingConfig, SelectionMode, SourceFolder
    from csliser.domain.planner import build_operation_plan
    from csliser.infrastructure import removal_receipt
    source = tmp_path/'frame_1.jpg'
    source.write_bytes(b'preserve despite disk error')
    item, = build_operation_plan(ProcessingConfig((SourceFolder(tmp_path, ('.jpg',)),), '1',
                  selection_mode=SelectionMode.FULL_RANGE, operation=FileOperation.DELETE)).operations
    def disk_error(*_args):
        raise OSError('controlled fsync failure')
    monkeypatch.setattr(removal_receipt.os, 'fsync', disk_error)
    with pytest.raises(OSError, match='fsync'):
        remove_verified_source(item)
    assert source.read_bytes() == b'preserve despite disk error'
    assert not discover_removal_receipts(tmp_path).operations


def test_cancel_during_source_quarantine_restores_original(tmp_path, monkeypatch):
    from threading import Event
    from csliser.application import file_transfer
    from csliser.application.use_cases import ExecuteTransferPlan
    from csliser.domain.models import FileOperation, ProcessingConfig, SelectionMode, SourceFolder
    from csliser.domain.planner import build_operation_plan
    source = tmp_path/'frame_1.jpg'
    source.write_bytes(b'preserve on cancellation')
    plan = build_operation_plan(ProcessingConfig((SourceFolder(tmp_path, ('.jpg',)),), '1',
                  selection_mode=SelectionMode.FULL_RANGE, operation=FileOperation.DELETE))
    cancel = Event()
    rename = file_transfer._rename_new
    def cancel_after_rename(a, b):
        rename(a, b)
        if a == source:
            cancel.set()
    monkeypatch.setattr(file_transfer, '_rename_new', cancel_after_rename)
    result = ExecuteTransferPlan().execute(plan, cancelled=cancel.is_set)
    assert result.cancelled and not result.errors and result.completed == 0
    assert source.read_bytes() == b'preserve on cancellation'
    assert RecoverRemovals().execute(discover_removal_receipts(tmp_path)).ok
