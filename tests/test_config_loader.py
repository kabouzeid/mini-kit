import textwrap
from pathlib import Path

from mini.config import load_config, load_merged_config


def _write(path: Path, code: str):
    path.write_text(textwrap.dedent(code))


def test_parent_precedence(tmp_path):
    """
    parent1  -> parent2  -> child
       lr=0.1    lr=0.01    batch_size=64
    Expect: lr from parent2, plus optim from parent1, plus batch_size.
    """
    p1 = tmp_path / "parent1.py"
    _write(
        p1,
        """
        config = {"lr": 0.1, "optim": "sgd"}
        """,
    )

    p2 = tmp_path / "parent2.py"
    _write(
        p2,
        """
        parents = ["parent1.py"]
        config = {"lr": 0.01}
        """,
    )

    child = tmp_path / "child.py"
    _write(
        child,
        """
        parents = ["parent2.py"]
        config = {"batch_size": 64}
        """,
    )

    cfg = load_config(child)
    assert cfg == {"lr": 0.01, "optim": "sgd", "batch_size": 64}


def test_key_deletion(tmp_path):
    """
    Child declares delete = ["model.dropout"].
    """
    parent = tmp_path / "parent.py"
    _write(
        parent,
        """
        config = {"model": {"name": "resnet", "dropout": 0.5}}
        """,
    )

    child = tmp_path / "child.py"
    _write(
        child,
        """
        parents = ["parent.py"]
        delete  = ["model.dropout"]
        config  = {}
        """,
    )

    cfg = load_config(child)
    assert cfg == {"model": {"name": "resnet"}}


def test_key_deletion2(tmp_path):
    """
    Child declares delete = ["model.name"] and also overrides it in the config.
    """
    parent = tmp_path / "parent.py"
    _write(
        parent,
        """
        config = {"model": {"name": "resnet"}}
        """,
    )

    child = tmp_path / "child.py"
    _write(
        child,
        """
        parents = ["parent.py"]
        delete  = ["model.name"]
        config  = {"model": {"name": "vit"}}
        """,
    )

    cfg = load_config(child)
    assert cfg == {"model": {"name": "vit"}}


def test_load_merged_config_order(tmp_path):
    """
    Earlier paths should be overridden by later ones.
    """
    a = tmp_path / "a.py"
    _write(a, "config = {'a': 1, 'b': 2}")

    b = tmp_path / "b.py"
    _write(b, "config = {'b': 3, 'c': 4}")

    merged = load_merged_config([a, b])
    assert merged == {"a": 1, "b": 3, "c": 4}
