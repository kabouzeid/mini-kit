import io
import tarfile

import pytest
from mini.itar.indexed_tar import IndexedTar


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


@pytest.fixture
def tar_and_files(tmp_path):
    files = {
        "a.txt": b"hello",
        "b.txt": b"world",
        "dir/c.txt": b"!",
    }
    tar_bytes = make_tar_bytes(files)
    return tar_bytes, files


def test_indexed_tar_basic(tar_and_files):
    tar_bytes, files = tar_and_files
    itar = IndexedTar(tar_bytes)
    assert set(itar.keys()) == set(files)
    assert len(itar) == len(files)
    for name, content in files.items():
        with itar[name] as f:
            assert f.read() == content
        assert name in itar
    assert list(itar) == list(files)
    assert dict(itar.items()).keys() == set(files)
    itar.close()


def test_indexed_tar_with_links_and_symlinks(tmp_path):
    # Create files and add a hard link and a symlink
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        # Regular file
        info_a = tarfile.TarInfo("a.txt")
        data_a = b"hello"
        info_a.size = len(data_a)
        tf.addfile(info_a, io.BytesIO(data_a))

        # Hard link to a.txt
        info_hard = tarfile.TarInfo("hardlink.txt")
        info_hard.type = tarfile.LNKTYPE
        info_hard.linkname = "a.txt"
        tf.addfile(info_hard)

        # Symlink to a.txt
        info_sym = tarfile.TarInfo("symlink.txt")
        info_sym.type = tarfile.SYMTYPE
        info_sym.linkname = "a.txt"
        tf.addfile(info_sym)

    buf.seek(0)
    itar = IndexedTar(buf)
    # All keys present
    assert set(itar.keys()) == {"a.txt", "hardlink.txt", "symlink.txt"}
    # Hard link and symlink resolve to same content as a.txt
    assert itar["hardlink.txt"].read() == b"hello"
    assert itar["symlink.txt"].read() == b"hello"
    # Info objects
    info_hard = itar.info("hardlink.txt")
    info_sym = itar.info("symlink.txt")
    assert info_hard.islnk()
    assert info_sym.issym()
    # Verify index does not raise
    for name in ["a.txt", "hardlink.txt", "symlink.txt"]:
        itar.verify_index(name)
    itar.close()


def test_indexed_tar_info(tar_and_files):
    tar_bytes, files = tar_and_files
    itar = IndexedTar(tar_bytes)
    for name in files:
        info = itar.info(name)
        assert info.name == name
        assert info.size == len(files[name])
    itar.close()


def test_indexed_tar_verify_index(tar_and_files):
    tar_bytes, files = tar_and_files
    itar = IndexedTar(tar_bytes)
    for name in files:
        itar.verify_index(name)
    itar.close()


def test_indexed_tar_verify_index_raises(tar_and_files):
    tar_bytes, files = tar_and_files
    itar = IndexedTar(tar_bytes, index={k: (0, 512, 0, None) for k, v in files.items()})
    for name in files:
        with pytest.raises(ValueError):
            itar.verify_index(name)
    itar.close()


def test_indexed_tar_context_manager(tar_and_files):
    tar_bytes, files = tar_and_files
    with IndexedTar(tar_bytes) as itar:
        for name in files:
            assert itar[name].read() == files[name]


def test_indexed_tar_build_index(tmp_path):
    files = {
        "x.txt": b"abc",
        "y.txt": b"defg",
    }
    tar_bytes = make_tar_bytes(files)
    itar = IndexedTar(tar_bytes)
    assert itar.index == {"x.txt": (0, 512, 3, None), "y.txt": (1024, 1536, 4, None)}
    assert set(itar.keys()) == set(files)
    for name in files:
        assert itar[name].read() == files[name]
    itar.close()


def test_indexed_tar_missing_key(tar_and_files):
    tar_bytes, _ = tar_and_files
    itar = IndexedTar(tar_bytes)
    with pytest.raises(KeyError):
        _ = itar["notfound.txt"]
    itar.close()
