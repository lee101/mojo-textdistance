from __future__ import annotations

from collections import Counter
import random

import pytest

import textdistance as mojo


@pytest.mark.parametrize(
    "left,right",
    [
        ("", "abc"),
        ("abcdef", "acf"),
        ("XMJYAUZ", "MZJAWXU"),
        ("dbbdddabc", "cacc"),
        ("banana", "ananas"),
        ("東京京都", "京都大阪"),
    ],
)
def test_lcs_algorithms_match_upstream(upstream, left, right):
    for class_name in ("LCSSeq", "LCSStr"):
        ours = getattr(mojo, class_name)(external=False)
        theirs = getattr(upstream, class_name)(external=False)
        assert ours(left, right) == theirs(left, right)
        assert ours.similarity(left, right) == theirs.similarity(left, right)
        assert ours.distance(left, right) == theirs.distance(left, right)


def test_lcs_randomized_tie_breaking(upstream):
    rng = random.Random(17)
    ours = mojo.LCSStr(external=False)
    theirs = upstream.LCSStr(external=False)
    for _ in range(150):
        left = "".join(rng.choices("abcd", k=rng.randrange(20)))
        right = "".join(rng.choices("abcd", k=rng.randrange(20)))
        assert ours(left, right) == theirs(left, right)


def test_lcsstr_short_dispatch_boundary(upstream):
    ours = mojo.LCSStr(external=False)
    theirs = upstream.LCSStr(external=False)
    for size in (31, 32):
        left = "a" * (size - 3) + "bcd"
        right = "x" + left[1:-2] + "dc"
        assert ours(left, right) == theirs(left, right)


@pytest.mark.parametrize("sequence_type", [str, bytes])
def test_lcsstr_short_tie_breaking_and_no_match(upstream, sequence_type):
    ours = mojo.LCSStr(external=False)
    theirs = upstream.LCSStr(external=False)
    for left, right in [("ab12cd", "cd34ab"), ("abc", "XYZ")]:
        left = sequence_type(left, "ascii") if sequence_type is bytes else left
        right = sequence_type(right, "ascii") if sequence_type is bytes else right
        assert ours(left, right) == theirs(left, right)
    assert ours("abc", b"abc") == theirs("abc", b"abc")


def test_lcsseq_custom_comparator(upstream):
    comparator = lambda left, right: left.lower() == right.lower()
    ours = mojo.LCSSeq(test_func=comparator, external=False)
    theirs = upstream.LCSSeq(test_func=comparator, external=False)
    assert ours("aBcDe", "ABxDE") == theirs("aBcDe", "ABxDE")


@pytest.mark.parametrize(
    "left,right",
    [
        ("abcd", "abdc"),
        ("GESTALT PATTERN MATCHING", "GESTALT PRACTICE"),
        ("dbbdddabc", "cacc"),
        ("", "abc"),
    ],
)
def test_ratcliff_obershelp_matches_upstream(upstream, left, right):
    ours = mojo.RatcliffObershelp(external=False)
    theirs = upstream.RatcliffObershelp(external=False)
    assert ours(left, right) == pytest.approx(theirs(left, right))


TOKEN_CLASSES = ["Jaccard", "Sorensen", "Tversky", "Overlap", "Cosine", "Bag"]


@pytest.mark.parametrize("class_name", TOKEN_CLASSES)
@pytest.mark.parametrize("qval", [1, 2, 3])
def test_token_metrics_match_upstream(upstream, class_name, qval):
    options = {"qval": qval, "external": False}
    ours = getattr(mojo, class_name)(**options)
    theirs = getattr(upstream, class_name)(**options)
    for left, right in [
        ("test", "text"),
        ("night", "nacht"),
        ("context", "contact"),
        ("aaaaab", "aaabbb"),
    ]:
        assert ours(left, right) == pytest.approx(theirs(left, right))


@pytest.mark.parametrize(
    "class_name", ["Jaccard", "Sorensen", "Tversky", "Overlap", "Cosine"]
)
def test_token_set_mode_and_normalized_api(upstream, class_name):
    ours = getattr(mojo, class_name)(as_set=True, external=False)
    theirs = getattr(upstream, class_name)(as_set=True, external=False)
    left, right = "mississippi", "impossible"
    assert ours(left, right) == pytest.approx(theirs(left, right))
    assert ours.distance(left, right) == pytest.approx(theirs.distance(left, right))
    assert ours.normalized_similarity(left, right) == pytest.approx(
        theirs.normalized_similarity(left, right)
    )


@pytest.mark.parametrize("as_set", [False, True])
def test_token_stats_ascii_and_simd_tail(upstream, as_set):
    ours = mojo.Jaccard(as_set=as_set, external=False)
    theirs = upstream.Jaccard(as_set=as_set, external=False)
    for left, right in [
        ("abz" * 50, "acy" * 50),
        ("ab\u0102" * 50, "ac\u0101" * 50),
        ("\U0001f600a\U0001f642" * 50, "\U0001f600b\U0001f642" * 50),
    ]:
        assert ours(left, right) == pytest.approx(theirs(left, right))


@pytest.mark.parametrize("as_set", [False, True])
def test_jaccard_short_dispatch_boundary(upstream, as_set):
    ours = mojo.Jaccard(as_set=as_set, external=False)
    theirs = upstream.Jaccard(as_set=as_set, external=False)
    for size in (127, 128):
        left = ("abc" * 100)[:size]
        right = ("abd" * 100)[:size]
        assert ours(left, right) == pytest.approx(theirs(left, right))


@pytest.mark.parametrize("qval", [1, 2])
@pytest.mark.parametrize("as_set", [False, True])
def test_jaccard_short_repeated_tokens(upstream, qval, as_set):
    ours = mojo.Jaccard(qval=qval, as_set=as_set, external=False)
    theirs = upstream.Jaccard(qval=qval, as_set=as_set, external=False)
    left = "aabbbbbcc"
    right = "aaaabbcdd"
    assert ours(left, right) == pytest.approx(theirs(left, right))


def test_counter_and_multi_sequence_token_parity(upstream):
    counters = (
        Counter({"a": 5, "b": 2}),
        Counter({"a": 3, "c": 4}),
        Counter({"a": 4, "b": 1}),
    )
    for class_name in ("Jaccard", "Sorensen", "Tversky", "Overlap", "Cosine", "Bag"):
        ours = getattr(mojo, class_name)(external=False)
        theirs = getattr(upstream, class_name)(external=False)
        assert ours(*counters) == pytest.approx(theirs(*counters))


def test_tversky_parameters_match_upstream(upstream):
    options = {"ks": [0.3, 0.7], "bias": 0.2, "as_set": True, "external": False}
    ours = mojo.Tversky(**options)
    theirs = upstream.Tversky(**options)
    assert ours("abracadabra", "alakazam") == pytest.approx(
        theirs("abracadabra", "alakazam")
    )


def test_tanimoto_and_aliases_match_upstream(upstream):
    left, right = "test", "text"
    assert mojo.Tanimoto(external=False)(left, right) == pytest.approx(
        upstream.Tanimoto(external=False)(left, right)
    )
    assert mojo.dice(left, right) == mojo.sorensen(left, right)
    assert mojo.sorensen_dice(left, right) == mojo.sorensen(left, right)
