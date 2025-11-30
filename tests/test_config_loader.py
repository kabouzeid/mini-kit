import textwrap
from pathlib import Path

from mini.config import load


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

    cfg = load(child)
    assert cfg == {"lr": 0.01, "optim": "sgd", "batch_size": 64}


def test_key_deletion(tmp_path):
    """
    Child deletes model.dropout.
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
        from mini.config import Delete
        parents = ["parent.py"]
        config  = {"model": {"dropout": Delete()}}
        """,
    )

    cfg = load(child)
    assert cfg == {"model": {"name": "resnet"}}


def test_key_replacement(tmp_path):
    """
    Child replaces model with a new dict.
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
        from mini.config import Replace
        parents = ["parent.py"]
        config  = {"model": Replace({"name": "vit", "activation": "relu"})}
        """,
    )

    cfg = load(child)
    assert cfg == {"model": {"name": "vit", "activation": "relu"}}


def test_load_multiple_configs_order(tmp_path):
    """
    Earlier paths should be overridden by later ones.
    """
    a = tmp_path / "a.py"
    _write(a, "config = {'a': 1, 'b': 2}")

    b = tmp_path / "b.py"
    _write(b, "config = {'b': 3, 'c': 4}")

    merged = load([a, b])
    assert merged == {"a": 1, "b": 3, "c": 4}


def test_param_injection(tmp_path):
    cfg_path = tmp_path / "cfg.py"
    _write(
        cfg_path,
        """
        variables =  {"steps": 1000}

        config = lambda v: {'steps': v.steps}
        """,
    )

    # Without injection we get defaults
    cfg_default = load(cfg_path)
    assert cfg_default == {"steps": 1000}

    # With injection we override usages consistently
    cfg_override = load(cfg_path, variables={"steps": 5000})
    assert cfg_override == {"steps": 5000}


def test_child_param_injection(tmp_path):
    cfg_path = tmp_path / "cfg.py"
    _write(
        cfg_path,
        """
        variables =  {"steps": 1000}

        config = lambda v: {'steps': v.steps}
        """,
    )
    child_cfg_path = tmp_path / "child_cfg.py"
    _write(
        child_cfg_path,
        f"""
        parents = ['{cfg_path}']
        variables =  {{"steps": 5000}}

        config = lambda v: {{"steps": v.steps}}
        """,
    )

    # Without injection we get defaults
    cfg_default = load(cfg_path)
    assert cfg_default == {"steps": 1000}

    # With child injection we override usages consistently
    cfg_override = load(child_cfg_path)
    assert cfg_override == {"steps": 5000}


def test_child_param_inheritance(tmp_path):
    cfg_path = tmp_path / "cfg.py"
    _write(
        cfg_path,
        """
        variables =  {"steps": 1000}

        config = lambda v: {'steps': v.steps}
        """,
    )
    child_cfg_path = tmp_path / "child_cfg.py"
    _write(
        child_cfg_path,
        f"""
        parents = ['{cfg_path}']
        config = lambda v: {{"warmup_steps": int(v.steps * 0.1)}}
        """,
    )

    cfg_default = load(cfg_path)
    assert cfg_default == {"steps": 1000}

    # The child inherits the parent's default for steps
    cfg_override = load(child_cfg_path)
    assert cfg_override == {"steps": 1000, "warmup_steps": 100}


def test_variables_nested(tmp_path):
    cfg_path = tmp_path / "cfg.py"
    _write(
        cfg_path,
        """
        variables =  {"training": {"steps": 200, "device": "cuda"}}

        config = lambda v: {
            "steps": v.training.steps,
            "device": v.training.device,
        }
        """,
    )

    cfg = load(cfg_path)
    assert cfg == {"steps": 200, "device": "cuda"}


def test_complex_child_param_inheritance(tmp_path):
    parent_cfg_paths = [tmp_path / f"{p}.py" for p in ["1", "2", "3"]]
    for p_path in parent_cfg_paths:
        grandparent_cfg_paths = [
            tmp_path / f"{p_path.stem}_{g}.py" for g in ["X", "Y", "Z"]
        ]
        for gp_path in grandparent_cfg_paths:
            vairables = (
                {"steps": 100, "warmup_steps": 100} if gp_path.stem == "3_X" else {}
            )
            _write(
                gp_path,
                f"""
                variables = {vairables}

                config = lambda v: {{"steps_{gp_path.stem}": v.steps}}
                """,
            )
        _write(
            p_path,
            f"""
            parents = {list(map(str, grandparent_cfg_paths))}
            variables =  {{"steps": {int(p_path.stem) * 1000}, "warmup_steps": 100, "extra": False}}

            config = lambda v: {{
                "steps_{p_path.stem}": v.steps,
                "warmup_steps_{p_path.stem}": v.warmup_steps,
                "extra_{p_path.stem}": v.extra,
            }}
            """,
        )

    base_cfg_path = tmp_path / "base.py"
    _write(
        base_cfg_path,
        f"""
        parents = {list(map(str, parent_cfg_paths))}
        config = lambda v: {{
            "steps": v.steps,
            "warmup_steps": v.warmup_steps,
            "extra": v.extra,
        }}
        """,
    )

    cfg = load(base_cfg_path, {"extra": True})
    assert cfg == {
        "extra": True,
        "steps": 3000,
        "warmup_steps": 100,
        "extra_1": True,
        "steps_1": 3000,
        "warmup_steps_1": 100,
        "steps_1_X": 3000,
        "steps_1_Y": 3000,
        "steps_1_Z": 3000,
        "extra_2": True,
        "steps_2": 3000,
        "warmup_steps_2": 100,
        "steps_2_X": 3000,
        "steps_2_Y": 3000,
        "steps_2_Z": 3000,
        "extra_3": True,
        "steps_3": 3000,
        "warmup_steps_3": 100,
        "steps_3_X": 3000,
        "steps_3_Y": 3000,
        "steps_3_Z": 3000,
    }
