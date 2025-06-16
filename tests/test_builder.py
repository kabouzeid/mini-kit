from types import SimpleNamespace

import pytest

from mini.builder import REGISTRY, Registry, build_from_cfg


@REGISTRY.register()
class Leaf:
    def __init__(self, value):
        self.value = value


@REGISTRY.register()
class Node:
    def __init__(self, children):
        self.children = children


@REGISTRY.register()
class Tree:
    def __init__(self, root, name):
        self.root = root
        self.name = name


def test_simple_instantiation():
    cfg = {"type": "Leaf", "value": 123}
    obj = build_from_cfg(cfg, registry=REGISTRY)
    assert isinstance(obj, Leaf)
    assert obj.value == 123


def test_nested_instantiation():
    cfg = {
        "type": "Tree",
        "name": "MyTree",
        "root": {
            "type": "Node",
            "children": [
                {"type": "Leaf", "value": 1},
                {"type": "Leaf", "value": 2},
            ],
        },
    }
    tree = build_from_cfg(cfg, registry=REGISTRY, recursive=True)
    assert isinstance(tree, Tree)
    assert tree.name == "MyTree"
    assert isinstance(tree.root, Node)
    assert all(isinstance(child, Leaf) for child in tree.root.children)
    assert tree.root.children[0].value == 1


def test_tuple_support():
    cfg = {
        "type": "Node",
        "children": ({"type": "Leaf", "value": 10}, {"type": "Leaf", "value": 20}),
    }
    node = build_from_cfg(cfg, registry=REGISTRY, recursive=True)
    assert isinstance(node.children, tuple)
    assert isinstance(node.children[0], Leaf)


def test_no_recursion():
    cfg = {"type": "Node", "children": [{"type": "Leaf", "value": 1}]}
    node = build_from_cfg(cfg, registry=REGISTRY, recursive=False)
    assert isinstance(node.children[0], dict)  # no recursive instantiation


def test_module_path_instantiation():
    cfg = {"type": "types.SimpleNamespace", "x": 42}
    obj = build_from_cfg(cfg, recursive=True)  # no registry
    assert isinstance(obj, SimpleNamespace)
    assert obj.x == 42


def test_missing_type_key():
    with pytest.raises(AssertionError):
        build_from_cfg({"x": 1})


def test_invalid_class_path():
    with pytest.raises(ModuleNotFoundError):
        build_from_cfg({"type": "nonexistent.Module", "x": 1})


def test_ignore_registry():
    with pytest.raises(ValueError):
        build_from_cfg({"type": "Leaf", "value": 123}, registry=None)


def test_custom_registry():
    custom_registry = Registry()

    @custom_registry.register("Alias")
    class Dummy:
        def __init__(self, x):
            self.x = x

    cfg = {"type": "Alias", "x": 42}
    obj = build_from_cfg(cfg, registry=custom_registry)
    assert isinstance(obj, Dummy)
    assert obj.x == 42
