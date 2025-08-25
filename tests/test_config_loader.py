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


def test_param_injection(tmp_path):
    cfg_path = tmp_path / "cfg.py"
    _write(
        cfg_path,
        """
        from mini.config import param
        total_steps = param('total_steps', 1000)
        warmup_fraction = 0.1
        config = {
            'total_steps': total_steps,
            'warmup_steps': int(total_steps * warmup_fraction),
        }
        """,
    )

    # Without injection we get defaults
    cfg_default = load_config(cfg_path)
    assert cfg_default == {"total_steps": 1000, "warmup_steps": 100}

    # With injection we override usages consistently
    cfg_override = load_config(cfg_path, params={"total_steps": 5000})
    assert cfg_override == {"total_steps": 5000, "warmup_steps": 500}


def test_child_param_injection(tmp_path):
    cfg_path = tmp_path / "cfg.py"
    _write(
        cfg_path,
        """
        from mini.config import param
        total_steps = param('total_steps', 1000)
        warmup_fraction = 0.1
        config = {
            'total_steps': total_steps,
            'warmup_steps': int(total_steps * warmup_fraction),
        }
        """,
    )
    child_cfg_path = tmp_path / "child_cfg.py"
    _write(
        child_cfg_path,
        f"""
        params = {{'total_steps': 5000}}
        parents = ['{cfg_path}']
        """,
    )

    # Without injection we get defaults
    cfg_default = load_config(cfg_path)
    assert cfg_default == {"total_steps": 1000, "warmup_steps": 100}

    # With child injection we override usages consistently
    cfg_override = load_config(child_cfg_path)
    assert cfg_override == {"total_steps": 5000, "warmup_steps": 500}
