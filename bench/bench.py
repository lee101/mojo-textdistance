"""Benchmark Mojo kernels against textdistance 4.x on identical inputs."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import math
import os
import pathlib
import platform
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import textdistance as mojo  # noqa: E402


def load_upstream():
    distribution = importlib.metadata.distribution("textdistance")
    package = pathlib.Path(distribution.locate_file("textdistance"))
    spec = importlib.util.spec_from_file_location(
        "benchmark_upstream_textdistance",
        package / "__init__.py",
        submodule_search_locations=[str(package)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def best_time(function, repeats: int = 3) -> float:
    best = math.inf
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - started)
    return best


def batched_time(function, iterations: int) -> float:
    return best_time(lambda: [function() for _ in range(iterations)]) / iterations


def random_text(length: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choices("abcdefghijklmnopqrstuvwxyz ", k=length))


def mutate(text: str, every: int) -> str:
    values = list(text)
    for index in range(every // 2, len(values), every):
        values[index] = "x" if values[index] != "x" else "y"
    return "".join(values)


def cpu_name() -> str:
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main() -> None:
    upstream = load_upstream()
    cases = []

    first = random_text(2_000_000, 1)
    second = mutate(first, 97)
    cases.append(
        (
            "Hamming",
            "2,000,000 chars",
            mojo.Hamming(external=False),
            upstream.Hamming(external=False),
            first,
            second,
        )
    )

    first = random_text(2_000, 2)
    second = mutate(first, 23)
    cases.append(
        (
            "Levenshtein",
            "2,000 x 2,000 chars",
            mojo.Levenshtein(external=False),
            upstream.Levenshtein(external=False),
            first,
            second,
        )
    )

    first = random_text(900, 3)
    second = mutate(first, 17)
    cases.append(
        (
            "Damerau-Levenshtein",
            "900 x 900 chars",
            mojo.DamerauLevenshtein(external=False),
            upstream.DamerauLevenshtein(external=False),
            first,
            second,
        )
    )

    first = random_text(10_000, 4)
    second = mutate(first, 251)
    cases.append(
        (
            "Jaro-Winkler",
            "10,000 chars",
            mojo.JaroWinkler(external=False),
            upstream.JaroWinkler(external=False),
            first,
            second,
        )
    )

    first = random_text(1_100, 5)
    second = mutate(first, 13)
    cases.append(
        (
            "LCS sequence",
            "1,100 x 1,100 chars",
            mojo.LCSSeq(external=False).similarity,
            upstream.LCSSeq(external=False).similarity,
            first,
            second,
        )
    )

    first = random_text(1_100, 6)
    second = mutate(first, 19)
    cases.append(
        (
            "Needleman-Wunsch",
            "1,100 x 1,100 chars",
            mojo.NeedlemanWunsch(external=False),
            upstream.NeedlemanWunsch(external=False),
            first,
            second,
        )
    )

    first = random_text(1_000_000, 7)
    second = mutate(first, 29)
    cases.append(
        (
            "Jaccard",
            "1,000,000 chars",
            mojo.Jaccard(external=False),
            upstream.Jaccard(external=False),
            first,
            second,
        )
    )

    mojo.levenshtein("warm", "worm")
    print(f"Machine: {cpu_name()}, {platform.system()} {platform.machine()}")
    print(f"Python: {platform.python_version()}; upstream textdistance: {upstream.__version__}")
    print()
    print("| Metric | Input | Mojo | textdistance | Speedup |")
    print("|---|---:|---:|---:|---:|")
    for name, size, ours, theirs, left, right in cases:
        ours_result = ours(left, right)
        theirs_result = theirs(left, right)
        if isinstance(ours_result, float):
            if not math.isclose(ours_result, theirs_result, rel_tol=1e-12, abs_tol=1e-12):
                raise AssertionError(f"{name} result mismatch")
        elif ours_result != theirs_result:
            raise AssertionError(f"{name} result mismatch")
        mojo_seconds = best_time(lambda: ours(left, right))
        upstream_seconds = best_time(lambda: theirs(left, right))
        print(
            f"| {name} | {size} | {mojo_seconds * 1e3:.2f} ms | "
            f"{upstream_seconds * 1e3:.2f} ms | {upstream_seconds / mojo_seconds:.2f}x |"
        )

    if os.environ.get("MTD_PROFILE") != "1":
        return

    print()
    print("| Metric | Length | Mojo | textdistance | Speedup |")
    print("|---|---:|---:|---:|---:|")
    profile_metrics = [
        (
            "Hamming",
            mojo.Hamming(external=False),
            upstream.Hamming(external=False),
            [8, 64, 1024],
        ),
        (
            "Levenshtein",
            mojo.Levenshtein(external=False),
            upstream.Levenshtein(external=False),
            [8, 32, 128],
        ),
        (
            "Damerau-Levenshtein",
            mojo.DamerauLevenshtein(external=False),
            upstream.DamerauLevenshtein(external=False),
            [8, 32, 128],
        ),
        (
            "Jaro-Winkler",
            mojo.JaroWinkler(external=False),
            upstream.JaroWinkler(external=False),
            [8, 64, 512],
        ),
        (
            "LCS sequence",
            mojo.LCSSeq(external=False),
            upstream.LCSSeq(external=False),
            [8, 32, 128],
        ),
        (
            "LCS substring",
            mojo.LCSStr(external=False),
            upstream.LCSStr(external=False),
            [8, 64, 256],
        ),
        (
            "Needleman-Wunsch",
            mojo.NeedlemanWunsch(external=False),
            upstream.NeedlemanWunsch(external=False),
            [8, 32, 128],
        ),
        (
            "Smith-Waterman",
            mojo.SmithWaterman(external=False),
            upstream.SmithWaterman(external=False),
            [8, 32, 128],
        ),
        (
            "Jaccard",
            mojo.Jaccard(external=False),
            upstream.Jaccard(external=False),
            [8, 64, 4096],
        ),
        ("Prefix", mojo.Prefix(), upstream.Prefix(), [8, 64, 4096]),
        ("Postfix", mojo.Postfix(), upstream.Postfix(), [8, 64, 4096]),
    ]
    for metric_index, (name, ours, theirs, sizes) in enumerate(
        profile_metrics
    ):
        for size in sizes:
            left = random_text(size, 1000 + metric_index * 10 + size)
            right = mutate(left, max(3, size // 4))
            iterations = 100 if size <= 32 else 20 if size <= 128 else 3
            ours_result = ours(left, right)
            theirs_result = theirs(left, right)
            if isinstance(ours_result, float):
                if not math.isclose(
                    ours_result, theirs_result, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise AssertionError(f"{name} profile result mismatch")
            elif ours_result != theirs_result:
                raise AssertionError(f"{name} profile result mismatch")
            mojo_seconds = batched_time(lambda: ours(left, right), iterations)
            upstream_seconds = batched_time(lambda: theirs(left, right), iterations)
            print(
                f"| {name} | {size} | {mojo_seconds * 1e6:.2f} us | "
                f"{upstream_seconds * 1e6:.2f} us | "
                f"{upstream_seconds / mojo_seconds:.2f}x |"
            )


if __name__ == "__main__":
    main()
