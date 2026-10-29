"""Frozen code identity, relocation and import isolation without tensor libraries."""
from pathlib import Path

import pytest

from generals_arena.policies.code import load_policy_factory
from generals_arena.policies.code import policy_digest
from generals_arena.policies.code import snapshot_policy_code


def package(root, value):
    root.mkdir()
    (root/'__init__.py').write_text('')
    (root/'helper.py').write_text(f'VALUE = {value}\n')
    (root/'policy.py').write_text('from .helper import VALUE\ndef create_controller(): return VALUE\n')
    return root


def test_identical_code_is_portable_and_bytecode_does_not_change_identity(tmp_path):
    original = package(tmp_path/'original', 3)
    expected = policy_digest(original)
    copied = tmp_path/'copied'
    snapshot_policy_code(original, copied)
    assert policy_digest(copied) == expected
    assert load_policy_factory(original, 'policy:create_controller', expected)() == 3
    assert policy_digest(original) == expected
    assert load_policy_factory(copied, 'policy:create_controller', expected)() == 3
    (original/'helper.py').write_text('VALUE = 4\n')
    assert policy_digest(original) != expected
    assert load_policy_factory(copied, 'policy:create_controller', expected)() == 3
    with pytest.raises(ValueError, match='SHA256'):
        load_policy_factory(original, 'policy:create_controller', expected)


def test_different_versions_do_not_share_relative_imports(tmp_path):
    first = package(tmp_path/'first', 1)
    second = package(tmp_path/'second', 2)
    for root, expected in ((first, 1), (second, 2), (first, 1)):
        assert load_policy_factory(root, 'policy:create_controller', policy_digest(root))() == expected


def test_source_validation_and_entrypoint(tmp_path):
    root = package(tmp_path/'package', 1)
    for entry in ('../policy:create_controller', 'policy:missing', 'policy:VALUE'):
        with pytest.raises(ValueError):
            load_policy_factory(root, entry, policy_digest(root))
    (root/'weights.pt').write_bytes(b'not source')
    with pytest.raises(ValueError, match='unsupported'):
        policy_digest(root)
    (root/'weights.pt').unlink()
    (root/'link.py').symlink_to(root/'helper.py')
    with pytest.raises(ValueError, match='symlink'):
        policy_digest(root)


def test_copy_mutation_never_leaves_partial_source(tmp_path, monkeypatch):
    import generals_arena.policies.code as code
    root = package(tmp_path/'package', 1)
    copyfile = code.shutil.copyfile
    def mutate(source, target):
        result = copyfile(source, target)
        if Path(source).name == 'helper.py':
            Path(target).write_text('VALUE = 9\n')
        return result
    monkeypatch.setattr(code.shutil, 'copyfile', mutate)
    with pytest.raises(ValueError, match='changed'):
        snapshot_policy_code(root, tmp_path/'failed')
    assert not (tmp_path/'failed').exists()


def test_verified_source_is_executed_even_with_a_stale_bytecode_cache(tmp_path):
    import os
    import py_compile
    root = package(tmp_path/'package', 1)
    helper = root/'helper.py'
    stamp = helper.stat()
    py_compile.compile(str(helper), doraise=True)
    helper.write_text('VALUE = 2\n')  # Same size/mtime can fool timestamp-based .pyc validation.
    os.utime(helper, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert load_policy_factory(root, 'policy:create_controller', policy_digest(root))() == 2


def test_sourceless_bytecode_is_never_loaded_as_frozen_policy(tmp_path):
    import py_compile
    root = package(tmp_path/'package', 1)
    py_compile.compile(str(root/'policy.py'), cfile=str(root/'policy.pyc'), doraise=True)
    (root/'policy.py').unlink()
    with pytest.raises(ImportError, match='verified Python source'):
        load_policy_factory(root, 'policy:create_controller', policy_digest(root))
