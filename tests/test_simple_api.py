from __future__ import annotations

import pytest

import textdistance as mojo


@pytest.mark.parametrize(
    "class_name,left,right",
    [
        ("Prefix", "abcdef", "abcxyz"),
        ("Prefix", "abcd", "xyz"),
        ("Postfix", "xyzdef", "abcdef"),
        ("Postfix", "abc", "xyz"),
        ("Length", "abc", "abcdef"),
        ("Identity", "same", "same"),
        ("Identity", "same", "other"),
    ],
)
def test_simple_metrics_match_upstream(upstream, class_name, left, right):
    ours = getattr(mojo, class_name)()
    theirs = getattr(upstream, class_name)()
    assert ours(left, right) == theirs(left, right)
    assert ours.similarity(left, right) == theirs.similarity(left, right)
    assert ours.distance(left, right) == theirs.distance(left, right)


def test_prefix_and_postfix_sequence_results(upstream):
    for class_name in ("Prefix", "Postfix"):
        ours = getattr(mojo, class_name)()
        theirs = getattr(upstream, class_name)()
        assert ours([1, 2, 3], [1, 2, 4]) == theirs([1, 2, 3], [1, 2, 4])


@pytest.mark.parametrize("class_name", ["Prefix", "Postfix"])
def test_affix_simd_tail_and_dispatch_boundary(upstream, class_name):
    ours = getattr(mojo, class_name)()
    theirs = getattr(upstream, class_name)()
    for size in (255, 256, 263):
        if class_name == "Prefix":
            left = "a" * (size - 1) + "x"
            right = "a" * (size - 1) + "y"
        else:
            left = "x" + "a" * (size - 1)
            right = "y" + "a" * (size - 1)
        assert ours(left, right) == theirs(left, right)


def test_matrix_configuration_matches_upstream(upstream):
    options = {
        "mat": {("a", "b"): 7},
        "mismatch_cost": -2,
        "match_cost": 3,
        "symmetric": True,
    }
    ours = mojo.Matrix(**options)
    theirs = upstream.Matrix(**options)
    for values in (("a", "b"), ("b", "a"), ("x", "x"), ("x", "y")):
        assert ours(*values) == theirs(*values)


def test_module_instances_and_algorithm_imports():
    from textdistance.algorithms.edit_based import Levenshtein
    from textdistance.algorithms.token_based import Jaccard

    assert isinstance(mojo.levenshtein, Levenshtein)
    assert isinstance(mojo.jaccard, Jaccard)
    assert mojo.levenshtein("kitten", "sitting") == 3
    assert mojo.jaccard("test", "text") == pytest.approx(0.6)
