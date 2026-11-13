"""Build identity caching preserves recursive include resolution and invalidation."""
from hashlib import sha256
import os
from pathlib import Path

import pytest

from generals.backends._build import source_bytes, source_fingerprint


def test_recursive_fingerprint_reuses_reads_and_detects_header_changes(tmp_path, monkeypatch):
    source, header, nested = [tmp_path / name for name in ('main.cpp', 'header.h', 'nested.h')]
    source.write_text('#include "header.h"\n#include "nested.h"\nint main() {}')
    header.write_text('#include "nested.h"\n#include "main.cpp"\n')
    nested.write_text('one')
    expected = sha256(source_bytes(source)).hexdigest()
    original, reads = Path.read_bytes, []
    def read(path):
        reads.append(path)
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', read)
    assert source_fingerprint(source) == source_fingerprint(source) == expected
    assert reads == [source, header, nested]
    before = nested.stat()
    nested.write_text('two')
    os.utime(nested, ns=(before.st_atime_ns, before.st_mtime_ns))
    changed = source_fingerprint(source)
    assert changed != expected and changed == sha256(source_bytes(source)).hexdigest()
    replacement = tmp_path / 'replacement'
    replacement.write_text('six')
    replacement.replace(nested)
    assert source_fingerprint(source) != changed
    nested.unlink()
    with pytest.raises(FileNotFoundError):
        source_fingerprint(source)


def test_new_include_shadow_and_symlink_retarget_invalidate_graph(tmp_path):
    includes = tmp_path / 'include'
    includes.mkdir()
    source = tmp_path / 'main.cpp'
    source.write_text('#include "header.h"')
    (includes / 'header.h').write_text('fallback')
    def fingerprint():
        value = source_fingerprint(source, include_dirs=[includes])
        assert value == sha256(source_bytes(source, include_dirs=[includes])).hexdigest()
        return value
    before = fingerprint()
    header = tmp_path / 'header.h'
    header.write_text('local')
    assert fingerprint() != before
    header.unlink()
    assert fingerprint() == before
    header.symlink_to(includes / 'header.h')
    assert fingerprint() == before
    (includes / 'other.h').write_text('other')
    header.unlink()
    header.symlink_to(includes / 'other.h')
    assert fingerprint() != before


def test_source_mutation_during_read_is_not_cached(tmp_path, monkeypatch):
    source = tmp_path / 'source.cpp'
    source.write_text('old')
    original = Path.read_bytes
    def read(path):
        data = original(path)
        path.write_text('new')
        return data
    monkeypatch.setattr(Path, 'read_bytes', read)
    with pytest.raises(ValueError, match='changed during verification'):
        source_fingerprint(source)
    monkeypatch.setattr(Path, 'read_bytes', original)
    assert source_fingerprint(source) == sha256(b'new').hexdigest()


def test_symlink_to_identical_inode_still_changes_relative_includes(tmp_path):
    left, right = tmp_path / 'left', tmp_path / 'right'
    left.mkdir()
    right.mkdir()
    (left / 'main.cpp').write_text('#include "header.h"')
    os.link(left / 'main.cpp', right / 'main.cpp')
    (left / 'header.h').write_text('left')
    (right / 'header.h').write_text('right')
    alias = tmp_path / 'alias'
    alias.symlink_to(left, target_is_directory=True)
    before = source_fingerprint(alias / 'main.cpp')
    alias.unlink()
    alias.symlink_to(right, target_is_directory=True)
    after = source_fingerprint(alias / 'main.cpp')
    assert after != before and after == sha256(source_bytes(alias / 'main.cpp')).hexdigest()


def test_cycle_cannot_hide_an_earlier_file_mutation(tmp_path, monkeypatch):
    source, header = tmp_path / 'main.cpp', tmp_path / 'header.h'
    source.write_text('#include "header.h"\nold')
    header.write_text('#include "main.cpp"')
    original = Path.read_bytes
    def read(path):
        if path == header:
            source.write_text('#include "header.h"\nnew')
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', read)
    with pytest.raises(ValueError, match='changed during verification'):
        source_fingerprint(source)
