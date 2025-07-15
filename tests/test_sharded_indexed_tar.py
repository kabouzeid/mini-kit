import io
import tarfile

import pytest
from mini.itar.sharded_indexed_tar import ShardedIndexedTar


def make_tar_bytes(files):
    """Create an in-memory tarfile with given files (dict of name: bytes)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return buf


def make_sharded_tar_bytes(files):
    return [make_tar_bytes(f) for f in files]


@pytest.fixture
def sharded_tar_and_files(tmp_path):
    files = [
        {
            "a.txt": b"hello",
            "b.txt": b"world",
            "dir/c.txt": b"!",
        },
        {
            "c.txt": b"foo",
            "d.txt": b"bar",
            "dir/e.txt": b"baz",
        },
        {
            "f.txt": b"foo_f",
            "g.txt": b"bar_g",
            "dir2/h.txt": b"baz_h",
        },
    ]
    tar_bytes = make_sharded_tar_bytes(files)
    return tar_bytes, files


def test_sharded_indexed_tar_basic(sharded_tar_and_files):
    tar_bytes, files_ls = sharded_tar_and_files
    itar = ShardedIndexedTar(tar_bytes)
    assert set(itar.keys()) == set(f for files in files_ls for f in files)
    assert len(itar) == sum(len(files) for files in files_ls)
    for files in files_ls:
        for name, content in files.items():
            with itar[name] as f:
                assert f.read() == content
            assert name in itar
    assert list(itar) == sum((list(files) for files in files_ls), start=[])
    assert dict(itar.items()).keys() == set(f for files in files_ls for f in files)
    itar.close()


def test_sharded_indexed_tar_info(sharded_tar_and_files):
    tar_bytes, files_ls = sharded_tar_and_files
    itar = ShardedIndexedTar(tar_bytes)
    for files in files_ls:
        for name in files:
            info = itar.info(name)
            assert info.name == name
            assert info.size == len(files[name])
    itar.close()


def test_sharded_indexed_tar_verify_index(sharded_tar_and_files):
    tar_bytes, files_ls = sharded_tar_and_files
    itar = ShardedIndexedTar(tar_bytes)
    for files in files_ls:
        for name in files:
            itar.verify_index(name)
    itar.close()


def test_sharded_indexed_tar_verify_index_raises(sharded_tar_and_files):
    tar_bytes, files_ls = sharded_tar_and_files
    itar = ShardedIndexedTar(
        tar_bytes,
        indices=[
            {k: (0, 512, 0, None) for k, v in files.items()} for files in files_ls
        ],
    )
    for files in files_ls:
        for name in files:
            with pytest.raises(ValueError):
                itar.verify_index(name)
    itar.close()


def test_sharded_indexed_tar_context_manager(sharded_tar_and_files):
    tar_bytes, files_ls = sharded_tar_and_files
    with ShardedIndexedTar(tar_bytes) as itar:
        for files in files_ls:
            for name in files:
                assert itar[name].read() == files[name]


def test_sharded_indexed_tar_build_indices(tmp_path):
    files_ls = [
        {"x.txt": b"some", "y.txt": b"files"},
        {"a.txt": b"foo", "b.txt": b"bar"},
    ]
    tar_bytes = make_sharded_tar_bytes(files_ls)
    itar = ShardedIndexedTar(tar_bytes)
    assert itar.indices == [
        {"x.txt": (0, 512, 4, None), "y.txt": (1024, 1536, 5, None)},
        {"a.txt": (0, 512, 3, None), "b.txt": (1024, 1536, 3, None)},
    ]
    assert set(itar.keys()) == set(f for files in files_ls for f in files)
    for files in files_ls:
        for name in files:
            assert itar[name].read() == files[name]
    itar.close()


def test_indexed_tar_missing_key(sharded_tar_and_files):
    tar_bytes, _ = sharded_tar_and_files
    itar = ShardedIndexedTar(tar_bytes)
    with pytest.raises(KeyError):
        _ = itar["notfound.txt"]
    itar.close()
