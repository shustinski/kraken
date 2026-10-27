from tools.audit_reference_dataset import build_splits


def record(key, image, vector, issues=()):
    return dict(key=key, image_sha256=image, vector_sha256=vector, issues=list(issues))


def test_duplicate_bytes_never_leak_across_partitions_and_conflicting_labels_are_quarantined():
    records = [record(str(i), str(i), str(i)) for i in range(10)]
    records[8] = record('8', '0', '0')
    records[2] = record('2', 'conflict', 'label-a')
    records[9] = record('9', 'conflict', 'label-b')
    split, conflicts, duplicates = build_splits(records)
    assert '0' in split['train'] and '8' in split['train']
    assert not {'2', '9'} & set(sum(split.values(), []))
    assert conflicts == [dict(image_sha256='conflict', keys=['2', '9'],
        reason='identical_image_different_reference_bytes_requires_review')]
    assert duplicates == 2
    assert '6' in split['validation'] and '7' in split['validation']


def test_issues_and_missing_hash_are_excluded_without_becoming_negative_examples():
    split, conflicts, _ = build_splits([record('bad', 'bad', 'a', ['empty requires review']),
        dict(key='missing', issues=['decode error']), record('good', 'good', 'good')])
    assert sum(split.values(), []) == ['good']
    assert not conflicts
