# mojo-textdistance

`mojo-textdistance` is a Mojo implementation of the compute-heavy core of
[`textdistance`](https://pypi.org/project/textdistance/). It installs a Python
module named `textdistance`, preserves the upstream class and module-instance
style, and runs the quadratic or long-running loops in a compiled Mojo shared
library.

This is a focused port, not a claim to cover all 30-plus upstream algorithms.
The supported subset is parity-tested against `textdistance` 4.6.2.

## Coverage

| Family | Covered |
|---|---|
| Edit based | `Hamming`, `Levenshtein`, `DamerauLevenshtein` (restricted and unrestricted), `Jaro`, `JaroWinkler`, `NeedlemanWunsch`, `SmithWaterman` |
| Sequence based | `LCSSeq`, `LCSStr`, `RatcliffObershelp` |
| Token based | `Jaccard`, `Sorensen`, `Tversky`, `Overlap`, `Cosine`, `Tanimoto`, `Bag`, including `dice` and `sorensen_dice` aliases |
| Simple | `Prefix`, `Postfix`, `Length`, `Identity`, `Matrix` |

The usual `distance`, `similarity`, `normalized_distance`, and
`normalized_similarity` methods are available, along with upstream-style
callable module instances such as `textdistance.levenshtein` and
`textdistance.jaccard`. Strings, bytes, and ordinary sequences of hashable
items are supported. Unicode strings are compared by code point.

Custom `test_func` and `sim_func` callbacks are accepted where upstream accepts
them. Because a Python callback cannot safely be invoked from this C ABI, those
configurations use a correct Python fallback. Multi-sequence token operations
also fall back to Python; the two-sequence path is compiled.

Not covered are `MLIPNS`, `StrCmp95`, `Gotoh`, `MongeElkan`, phonetic metrics,
compression-based NCD metrics, and the remaining upstream algorithms. Those
names are deliberately absent instead of being misleading stubs.

## Install and run

The repository pins the tested Mojo nightly and manages Python dependencies
with Pixi:

```bash
pixi install
pixi run build
```

The build produces `dist/libmojo-textdistance.so`. The Pixi environment puts
the local `python/` directory on `PYTHONPATH`, so normal upstream imports work:

```bash
pixi run python - <<'PY'
import textdistance

print(textdistance.levenshtein("kitten", "sitting"))
print(textdistance.jaro_winkler("MARTHA", "MARHTA"))
print(textdistance.Jaccard(qval=2)("context", "contact"))
PY
```

This prints:

```text
3
0.9611111111111111
0.3333333333333333
```

Run validation and benchmarks with:

```bash
pixi run test
pixi run bench
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86-64, Python 3.13.14, Mojo
`1.1.0.dev2026081105`, and upstream `textdistance` 4.6.2. Each result is the
best of three runs after library warm-up. Upstream objects use
`external=False`, so this compares these kernels with upstream's own
implementations rather than an optionally installed third-party accelerator.
Inputs and return values are identical and the benchmark checks parity before
timing.

| Metric | Input | Mojo | textdistance | Speedup |
|---|---:|---:|---:|---:|
| Hamming | 2,000,000 chars | 3.25 ms | 577.63 ms | 177.60x |
| Levenshtein | 2,000 x 2,000 chars | 6.90 ms | 2492.74 ms | 361.33x |
| Damerau-Levenshtein | 900 x 900 chars | 2.99 ms | 1284.45 ms | 429.07x |
| Jaro-Winkler | 10,000 chars | 20.33 ms | 1373.79 ms | 67.58x |
| LCS sequence | 1,100 x 1,100 chars | 3.43 ms | 1133.42 ms | 330.62x |
| Needleman-Wunsch | 1,100 x 1,100 chars | 2.54 ms | 1412.27 ms | 556.73x |
| Jaccard | 1,000,000 chars | 2.20 ms | 102.05 ms | 46.40x |

These numbers describe this machine and workload, not a universal performance
guarantee. Input conversion and scratch allocation are included in the Mojo
timings.

## How it works

All kernels live in one Mojo compilation unit to avoid repeated compiler
startup cost. Python passes contiguous `uint32` token arrays. UTF-32 string
buffers remain zero-copy NumPy views through the FFI call; general sequences
are dense-coded so equal elements share an integer token.

The ctypes layer validates native dtype, contiguity, alignment, writability of
output buffers, and non-null pointers before passing integer addresses across
the C ABI. Exported Mojo functions reconstruct
`UnsafePointer[..., AnyOrigin[mut=True]]` values inside non-empty branches.
Python retains ownership of the input, result, and dynamic-programming scratch
arrays until each synchronous call returns, so the shared library performs no
allocation and exposes no cross-language lifetime to manage.

Scalar results return directly. Results such as an LCS substring location or
token-profile counts use small caller-owned result buffers. The Python layer
then reconstructs upstream-compatible strings, lists, and metric objects.

Prefix and postfix scans use native-width SIMD with scalar tails. Token
histogram initialization and reductions are also SIMD-vectorized, and sparse
Unicode inputs are compacted before allocating histogram scratch.

Short Jaccard calls compute intersection and union totals without allocating
temporary result counters. Short LCS substring calls use descending substring
search, preserving the upstream earliest-match tie break without constructing a
`SequenceMatcher`.

No GPU or threaded path is included. The linear scans and histogram kernels are
memory-bound, while the dynamic-programming and greedy matching kernels have
loop-carried dependencies. The added transfer, launch, and coordination costs
have no demonstrated benefit for the workloads in this repository.
