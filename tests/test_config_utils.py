import pytest
from mini.config.config import (
    apply_overrides,
    infer_type,
    parse_key_path,
    set_nested,
)

### --- parse_key_path tests --- ###


@pytest.mark.parametrize(
    "input_path, expected",
    [
        ("a.b.c", ["a", "b", "c"]),
        ("a[0].b", ["a", 0, "b"]),
        ("x[1][2].y", ["x", 1, 2, "y"]),
        ("x[-1].y", ["x", -1, "y"]),
        ("a", ["a"]),
        ("a[0]", ["a", 0]),
    ],
)
def test_parse_key_path(input_path, expected):
    assert parse_key_path(input_path) == expected


### --- infer_type tests --- ###


@pytest.mark.parametrize(
    "input_val, expected",
    [
        ("123", 123),
        ("3.14", 3.14),
        ("3e-4", 3e-4),
        ("True", True),
        ("None", None),
        ("[1, 2]", [1, 2]),
        ("{'x': 5}", {"x": 5}),
        ("{'x': [{'a': 1}, {'b': 2}]}", {"x": [{"a": 1}, {"b": 2}]}),
        ("'hello'", "hello"),
        ("unquoted_string", "unquoted_string"),  # fallback
    ],
)
def test_infer_type(input_val, expected):
    assert infer_type(input_val) == expected


### --- set_nested tests --- ###


def test_set_nested_dict():
    cfg = {}
    set_nested(cfg, ["a", "b", "c"], 42)
    assert cfg == {"a": {"b": {"c": 42}}}


def test_set_nested_list():
    cfg = {}
    set_nested(cfg, ["a", "b", 0, "x"], "hi")
    assert cfg == {"a": {"b": [{"x": "hi"}]}}


def test_set_nested_expand_list():
    cfg = []
    set_nested(cfg, [2], "foo")
    assert cfg == [None, None, "foo"]


def test_set_nested_last_list_index():
    cfg = ["a", "b"]
    set_nested(cfg, [-1], "c")
    assert cfg == ["a", "c"]


### --- apply_overrides tests --- ###


def test_apply_overrides_basic():
    cfg = {"a": {"b": 1}}
    overrides = ["a.b=42"]
    updated = apply_overrides(cfg, overrides)
    assert updated == {"a": {"b": 42}}


def test_apply_overrides_nested_list():
    cfg = {}
    overrides = ["x.y[0].name=conv", "x.y[1].attrs.out_channels=64"]
    updated = apply_overrides(cfg, overrides)
    assert updated == {"x": {"y": [{"name": "conv"}, {"attrs": {"out_channels": 64}}]}}


def test_apply_overrides_mix_types():
    cfg = {}
    overrides = ["flag=True", "threshold=0.75", "num_layers=5", "name=model_v1"]
    updated = apply_overrides(cfg, overrides)
    assert updated == {
        "flag": True,
        "threshold": 0.75,
        "num_layers": 5,
        "name": "model_v1",
    }


def test_apply_overrides_append():
    cfg = {"existing_list": ["a", "b"], "nested": {"items": []}}
    overrides = [
        "existing_list+=c",
        "new_list+=1",
        "new_list+=2",
        "new_list+=3",
        "nested.items+=foo",
        "nested.items+=bar",
    ]
    updated = apply_overrides(cfg, overrides)
    assert updated == {
        "existing_list": ["a", "b", "c"],
        "new_list": [1, 2, 3],
        "nested": {"items": ["foo", "bar"]},
    }


def test_delete_dict_key():
    cfg = {"a": {"b": {"c": 123, "d": 456}}}
    overrides = ["a.b.c!="]
    updated = apply_overrides(cfg, overrides)
    assert updated == {"a": {"b": {"d": 456}}}


def test_delete_list_index():
    cfg = {"layers": ["conv1", "conv2", "conv3"]}
    overrides = ["layers[1]!="]
    updated = apply_overrides(cfg, overrides)
    assert updated == {"layers": ["conv1", "conv3"]}


def test_delete_list_value():
    cfg = {"tags": ["debug", "train", "final"]}
    overrides = ['tags-="train"']
    updated = apply_overrides(cfg, overrides)
    assert updated == {"tags": ["debug", "final"]}


def test_set_special_key():
    cfg = {"a": {"b": 1}}
    overrides = ["a.*=[1,2,3]"]
    updated = apply_overrides(cfg, overrides)
    assert updated == {"a": {"*": [1, 2, 3], "b": 1}}


def test_combined_deletes_and_adds():
    cfg = {
        "model": {
            "layers": ["conv", "bn", "relu"],
            "dropout": 0.5,
        },
        "tags": ["baseline"],
    }
    overrides = [
        "model.dropout!=",  # delete dict key
        "model.layers[1]!=",  # delete list index
        "model.layers+=maxpool",  # append
        "tags+=debug",  # append
        'tags-="baseline"',  # remove value
    ]
    updated = apply_overrides(cfg, overrides)
    assert updated == {
        "model": {"layers": ["conv", "relu", "maxpool"]},
        "tags": ["debug"],
    }
