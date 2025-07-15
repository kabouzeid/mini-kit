import io
import tarfile

import pytest
from mini.itar.sitar import SITar


def make_tar_file(path, files):
    """Create a tar file at `path` with given files (dict of name: bytes)."""
    with tarfile.open(path, "w") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


@pytest.fixture
def sharded_tar_files(tmp_path):
    files_ls = [
        {"a.txt": b"hello", "b.txt": b"world"},
        {"c.txt": b"foo", "d.txt": b"bar"},
    ]
    tar_paths = []
    for i, files in enumerate(files_ls):
        tar_path = tmp_path / f"shard_{i}.tar"
        make_tar_file(tar_path, files)
        tar_paths.append(tar_path)
    return tar_paths, files_ls


def test_sitar_save_and_load(tmp_path, sharded_tar_files):
    tar_paths, files_ls = sharded_tar_files
    sitar_path = tmp_path / "archive.sitar"
    # Save SITar archive
    SITar.save(sitar_path, tar_paths)
    # Load SITar archive
    sitar = SITar(sitar_path)
    # Check keys and contents
    all_files = set(f for files in files_ls for f in files)
    assert set(sitar.keys()) == all_files
    for files in files_ls:
        for name, content in files.items():
            with sitar[name] as f:
                assert f.read() == content
    sitar.close()


def test_sitar_save_with_indices(tmp_path, sharded_tar_files):
    tar_paths, files_ls = sharded_tar_files
    sitar_path = tmp_path / "archive_with_indices.sitar"
    # Build indices manually
    indices = [None, None]
    SITar.save(sitar_path, tar_paths, indices=indices)
    sitar = SITar(sitar_path)
    for files in files_ls:
        for name, content in files.items():
            with sitar[name] as f:
                assert f.read() == content
    sitar.close()


def test_sitar_relative_paths(tmp_path, sharded_tar_files):
    tar_paths, files_ls = sharded_tar_files
    # Move shards to a subdir
    subdir = tmp_path / "shards"
    subdir.mkdir()
    new_tar_paths = []
    for tar_path in tar_paths:
        new_path = subdir / tar_path.name
        tar_path.rename(new_path)
        new_tar_paths.append(new_path)
    sitar_path = tmp_path / "archive_rel.sitar"
    SITar.save(sitar_path, new_tar_paths)
    sitar = SITar(sitar_path)
    for files in files_ls:
        for name, content in files.items():
            with sitar[name] as f:
                assert f.read() == content
    sitar.close()
