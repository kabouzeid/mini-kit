import math

import pytest
from mini.trainer.utils import key_average


def test_key_average_simple():
    dicts = [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
        {"a": 5, "b": 6},
    ]
    result = key_average(dicts)
    assert result == {"a": 3.0, "b": 4.0}


def test_key_average_missing_keys():
    dicts = [
        {"a": 1},
        {"a": 3, "b": 4},
        {"b": 6},
    ]
    result = key_average(dicts)
    assert result["a"] == 2.0
    assert result["b"] == 5.0


def test_key_average_nested_dicts():
    dicts = [
        {"a": {"x": 1, "y": 2}, "b": 2},
        {"a": {"x": 3, "y": 4}, "b": 4},
        {"a": {"x": 5, "y": 6}, "b": 6},
    ]
    result = key_average(dicts)
    assert result == {"a": {"x": 3.0, "y": 4.0}, "b": 4.0}


def test_key_average_nested_missing_keys():
    dicts = [
        {"a": {"x": 1}},
        {"a": {"x": 3, "y": 4}},
        {"a": {"y": 6}},
    ]
    result = key_average(dicts)
    assert result["a"]["x"] == 2.0
    assert result["a"]["y"] == 5.0


def test_key_average_nan_values():
    dicts = [
        {"a": 1, "b": math.nan},
        {"a": 3, "b": 4},
        {"a": 5, "b": 6},
    ]
    result = key_average(dicts)
    assert result["a"] == 3.0
    assert result["b"] == 5.0


def test_key_average_all_nan():
    dicts = [
        {"a": math.nan},
        {"a": math.nan},
    ]
    result = key_average(dicts)
    assert math.isnan(result["a"])


def test_key_average_empty_list():
    result = key_average([])
    assert result == {}


def test_key_average_empty_dicts():
    result = key_average([{}, {}, {}])
    assert result == {}


def test_key_average_deeply_nested():
    dicts = [
        {"a": {"b": {"c": 1}}},
        {"a": {"b": {"c": 3}}},
        {"a": {"b": {"c": 5}}},
    ]
    result = key_average(dicts)
    assert result == {"a": {"b": {"c": 3.0}}}
