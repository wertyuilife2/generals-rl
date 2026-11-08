import os
from hashlib import sha256
import pytest
from generals_arena import util


def test_file_hash_reuse_detects_same_size_restored_mtime_and_replacement(tmp_path,monkeypatch):
    path=tmp_path/'weights.pt';path.write_bytes(b'first')
    original=util.sha256_file
    reads=[]
    def read(stream):
        reads.append(stream.name)
        return original(stream)
    monkeypatch.setattr(util,'sha256_file',read)
    before=path.stat()
    assert util.file_digest(path)==util.file_digest(path)==sha256(b'first').hexdigest()
    assert len(reads)==1
    path.write_bytes(b'other');os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
    assert util.file_digest(path)==sha256(b'other').hexdigest() and len(reads)==2
    other=tmp_path/'replace';other.write_bytes(b'third');other.replace(path)
    assert util.file_digest(path)==sha256(b'third').hexdigest() and len(reads)==3
    path.unlink()
    with pytest.raises(FileNotFoundError):util.file_digest(path)
