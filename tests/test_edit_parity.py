from __future__ import annotations

import random

import pytest

import textdistance as mojo


PAIRS = [
    ("", ""),
    ("", "abc"),
    ("a", "a"),
    ("a", "b"),
    ("kitten", "sitting"),
    ("Saturday", "Sunday"),
    ("MARTHA", "MARHTA"),
    ("DWAYNE", "DUANE"),
    ("café", "coffee"),
    ("東京", "京都"),
]


@pytest.mark.parametrize("left,right", PAIRS)
def test_hamming_matches_upstream(upstream, left, right):
    ours = mojo.Hamming(external=False)
    theirs = upstream.Hamming(external=False)
    assert ours(left, right) == theirs(left, right)


def test_hamming_options_and_multiple_sequences(upstream):
    cases = [
        ({"truncate": True}, ("abc", "ax")),
        ({"qval": 2}, ("abcdef", "abqdef")),
        ({}, ([1, 2, 3], [1, 4], [1, 2, 5])),
    ]
    for options, sequences in cases:
        ours = mojo.Hamming(external=False, **options)
        theirs = upstream.Hamming(external=False, **options)
        assert ours(*sequences) == theirs(*sequences)


def test_hamming_short_dispatch_boundary(upstream):
    ours = mojo.Hamming(external=False)
    theirs = upstream.Hamming(external=False)
    for size in (63, 64):
        left = "a" * size
        right = left[:-1] + "b"
        assert ours(left, right) == theirs(left, right)


@pytest.mark.parametrize("left,right", PAIRS)
def test_levenshtein_matches_upstream(upstream, left, right):
    assert mojo.Levenshtein(external=False)(left, right) == upstream.Levenshtein(
        external=False
    )(left, right)


def test_levenshtein_qgrams_sequences_and_custom_comparator(upstream):
    comparator = lambda left, right: left.lower() == right.lower()
    cases = [
        ({"qval": 2}, ("night", "nacht")),
        ({}, ([1, 2, 3, 4], [1, 3, 4])),
        ({"test_func": comparator}, ("AbCd", "abcd")),
    ]
    for options, sequences in cases:
        ours = mojo.Levenshtein(external=False, **options)
        theirs = upstream.Levenshtein(external=False, **options)
        assert ours(*sequences) == theirs(*sequences)


@pytest.mark.parametrize("restricted", [True, False])
def test_damerau_randomized_parity(upstream, restricted):
    rng = random.Random(91 + restricted)
    ours = mojo.DamerauLevenshtein(external=False, restricted=restricted)
    theirs = upstream.DamerauLevenshtein(external=False, restricted=restricted)
    for _ in range(120):
        left = "".join(rng.choices("abcd", k=rng.randrange(9)))
        right = "".join(rng.choices("abcd", k=rng.randrange(9)))
        assert ours(left, right) == theirs(left, right)


def test_damerau_restricted_and_unrestricted_differ(upstream):
    for restricted in (True, False):
        ours = mojo.DamerauLevenshtein(external=False, restricted=restricted)
        theirs = upstream.DamerauLevenshtein(external=False, restricted=restricted)
        assert ours("CA", "ABC") == theirs("CA", "ABC")
    assert mojo.DamerauLevenshtein(restricted=True)("CA", "ABC") == 3
    assert mojo.DamerauLevenshtein(restricted=False)("CA", "ABC") == 2


@pytest.mark.parametrize(
    "left,right",
    [
        ("MARTHA", "MARHTA"),
        ("DIXON", "DICKSONX"),
        ("JELLYFISH", "SMELLYFISH"),
        ("test", "test"),
        ("", "x"),
        ("abcvwxyz", "cabvwxyz"),
    ],
)
def test_jaro_family_matches_upstream(upstream, left, right):
    for class_name in ("Jaro", "JaroWinkler"):
        ours = getattr(mojo, class_name)(external=False)
        theirs = getattr(upstream, class_name)(external=False)
        assert ours(left, right) == pytest.approx(theirs(left, right))
        assert ours.distance(left, right) == pytest.approx(theirs.distance(left, right))


def test_jaro_options_match_upstream(upstream):
    options = {"long_tolerance": True, "qval": 2}
    left, right = "the quick brown fox", "the quick blown fax"
    ours = mojo.JaroWinkler(external=False, **options)
    theirs = upstream.JaroWinkler(external=False, **options)
    assert ours(left, right, prefix_weight=0.08) == pytest.approx(
        theirs(left, right, prefix_weight=0.08)
    )


def test_jaro_short_dispatch_boundary(upstream):
    ours = mojo.JaroWinkler(external=False, long_tolerance=True)
    theirs = upstream.JaroWinkler(external=False, long_tolerance=True)
    for size in (31, 32):
        left = "a" * (size - 3) + "bcd"
        right = "a" * (size - 3) + "bdc"
        assert ours(left, right) == pytest.approx(theirs(left, right))


@pytest.mark.parametrize("class_name", ["NeedlemanWunsch", "SmithWaterman"])
def test_alignment_defaults_match_upstream(upstream, class_name):
    ours = getattr(mojo, class_name)(external=False, gap_cost=0.75)
    theirs = getattr(upstream, class_name)(external=False, gap_cost=0.75)
    for left, right in PAIRS:
        assert ours(left, right) == pytest.approx(theirs(left, right))


@pytest.mark.parametrize("class_name", ["NeedlemanWunsch", "SmithWaterman"])
def test_alignment_custom_similarity_fallback(upstream, class_name):
    score = lambda left, right: 2.0 if left == right else -0.5
    ours = getattr(mojo, class_name)(
        external=False, gap_cost=1.25, sim_func=score
    )
    theirs = getattr(upstream, class_name)(
        external=False, gap_cost=1.25, sim_func=score
    )
    assert ours("GATTACA", "GCATGCU") == pytest.approx(
        theirs("GATTACA", "GCATGCU")
    )


def test_normalized_levenshtein_interface(upstream):
    ours = mojo.Levenshtein(external=False)
    theirs = upstream.Levenshtein(external=False)
    for left, right in PAIRS:
        assert ours.similarity(left, right) == theirs.similarity(left, right)
        assert ours.normalized_distance(left, right) == pytest.approx(
            theirs.normalized_distance(left, right)
        )
        assert ours.normalized_similarity(left, right) == pytest.approx(
            theirs.normalized_similarity(left, right)
        )
