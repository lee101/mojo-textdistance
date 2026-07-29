"""Python API and sequence preparation for the Mojo kernels."""

from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from itertools import islice, repeat, zip_longest
from math import log, prod
from typing import Any, Callable, Sequence

import numpy as np

from ._lib import address, lib

TestFunc = Callable[..., bool]
SimFunc = Callable[..., float]


def _ident(*elements: object) -> bool:
    try:
        return len(set(elements)) == 1
    except TypeError:
        return all(left == right for left, right in zip(elements, elements[1:]))


def _prepare(sequence: Sequence, qval: int | None):
    if not qval:
        return sequence.split()
    if qval == 1:
        return sequence
    return list(zip(*(sequence[index:] for index in range(qval))))


def _dense_pair(
    first: Sequence, second: Sequence, exact_vocab: bool = False
) -> tuple[np.ndarray, np.ndarray, int]:
    if isinstance(first, str) and isinstance(second, str):
        left = np.frombuffer(
            first.encode("utf-32-le", errors="surrogatepass"), dtype="<u4"
        )
        right = np.frombuffer(
            second.encode("utf-32-le", errors="surrogatepass"), dtype="<u4"
        )
        if exact_vocab:
            if first.isascii() and second.isascii():
                vocab = 128
            else:
                vocab = int(max(left.max(initial=0), right.max(initial=0))) + 1
                if vocab > max(1024, 4 * (len(left) + len(right))):
                    combined = np.concatenate((left, right))
                    _, inverse = np.unique(combined, return_inverse=True)
                    left = inverse[: len(left)].astype(np.uint32, copy=False)
                    right = inverse[len(left) :].astype(np.uint32, copy=False)
                    vocab = int(inverse.max(initial=0)) + 1
        else:
            vocab = 0x110000
        return left, right, vocab
    if isinstance(first, (bytes, bytearray)) and isinstance(
        second, (bytes, bytearray)
    ):
        left = np.frombuffer(bytes(first), dtype=np.uint8).astype(np.uint32)
        right = np.frombuffer(bytes(second), dtype=np.uint8).astype(np.uint32)
        return left, right, 256

    mapping: dict[Any, int] = {}
    unhashable: list[tuple[Any, int]] = []
    next_code = 0

    def code(value: Any) -> int:
        nonlocal next_code
        try:
            existing = mapping.get(value)
            if existing is not None:
                return existing
            if next_code >= 2**32:
                raise OverflowError("more than 2**32 distinct sequence elements")
            mapping[value] = next_code
        except TypeError:
            for known, existing in unhashable:
                if value == known:
                    return existing
            if next_code >= 2**32:
                raise OverflowError("more than 2**32 distinct sequence elements")
            unhashable.append((value, next_code))
        result = next_code
        next_code += 1
        return result

    left = np.fromiter(
        (code(value) for value in first), dtype=np.uint32, count=len(first)
    )
    right = np.fromiter(
        (code(value) for value in second), dtype=np.uint32, count=len(second)
    )
    return left, right, next_code


def _pair(first: Sequence, second: Sequence, qval: int | None):
    prepared_first = _prepare(first, qval)
    prepared_second = _prepare(second, qval)
    left, right, vocab = _dense_pair(prepared_first, prepared_second)
    return prepared_first, prepared_second, left, right, vocab


class Base:
    def __init__(self, qval: int = 1, external: bool = True) -> None:
        self.qval = qval
        self.external = external

    @staticmethod
    def maximum(*sequences: Sequence) -> float:
        return max(map(len, sequences))

    def distance(self, *sequences: Sequence) -> float:
        return self(*sequences)

    def similarity(self, *sequences: Sequence) -> float:
        return self.maximum(*sequences) - self.distance(*sequences)

    def normalized_distance(self, *sequences: Sequence) -> float:
        maximum = self.maximum(*sequences)
        return 0 if maximum == 0 else self.distance(*sequences) / maximum

    def normalized_similarity(self, *sequences: Sequence) -> float:
        return 1 - self.normalized_distance(*sequences)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.__dict__})"


class BaseSimilarity(Base):
    def distance(self, *sequences: Sequence) -> float:
        return self.maximum(*sequences) - self.similarity(*sequences)

    def similarity(self, *sequences: Sequence) -> float:
        return self(*sequences)


def _jaro_python(
    first: Sequence,
    second: Sequence,
    winklerize: bool,
    long_tolerance: bool,
    prefix_weight: float,
) -> float:
    search_range = max(len(first), len(second)) // 2 - 1
    search_range = max(search_range, 0)
    first_flags = [False] * len(first)
    second_flags = [False] * len(second)
    common = 0
    for i, left_value in enumerate(first):
        low = max(0, i - search_range)
        high = min(i + search_range, len(second) - 1)
        for j in range(low, high + 1):
            if not second_flags[j] and left_value == second[j]:
                first_flags[i] = second_flags[j] = True
                common += 1
                break
    if not common:
        return 0.0
    next_second = 0
    transpositions = 0
    for i, matched in enumerate(first_flags):
        if matched:
            while not second_flags[next_second]:
                next_second += 1
            if first[i] != second[next_second]:
                transpositions += 1
            next_second += 1
    transpositions //= 2
    weight = (
        common / len(first)
        + common / len(second)
        + (common - transpositions) / common
    ) / 3
    if not winklerize or weight <= 0.7:
        return weight
    prefix_limit = min(len(first), len(second), 4)
    prefix = 0
    while prefix < prefix_limit and first[prefix] == second[prefix]:
        prefix += 1
    if prefix:
        weight += prefix * prefix_weight * (1.0 - weight)
    if not long_tolerance or min(len(first), len(second)) <= 4:
        return weight
    if common <= prefix + 1 or 2 * common < min(len(first), len(second)) + prefix:
        return weight
    adjustment = (common - prefix - 1) / (
        len(first) + len(second) - prefix * 2 + 2
    )
    return weight + (1.0 - weight) * adjustment


class Hamming(Base):
    def __init__(
        self,
        qval: int = 1,
        test_func: TestFunc | None = None,
        truncate: bool = False,
        external: bool = True,
    ) -> None:
        self.qval = qval
        self.test_func = test_func or _ident
        self.truncate = truncate
        self.external = external

    def __call__(self, *sequences: Sequence[object]) -> int:
        prepared = [_prepare(sequence, self.qval) for sequence in sequences]
        if len(prepared) < 2:
            return 0
        if _ident(*prepared):
            return 0
        if not all(prepared):
            return max(map(len, prepared))
        if len(prepared) == 2 and self.test_func is _ident:
            if (
                max(map(len, prepared)) < 64
                and isinstance(prepared[0], (str, bytes))
                and isinstance(prepared[1], (str, bytes))
            ):
                zipper = zip if self.truncate else zip_longest
                return sum(left != right for left, right in zipper(*prepared))
            left, right, _ = _dense_pair(prepared[0], prepared[1])
            return int(
                lib().mtd_hamming(
                    address(left),
                    len(left),
                    address(right),
                    len(right),
                    int(self.truncate),
                )
            )
        zipper = zip if self.truncate else zip_longest
        return sum(not self.test_func(*values) for values in zipper(*prepared))


class Levenshtein(Base):
    def __init__(
        self,
        qval: int = 1,
        test_func: TestFunc | None = None,
        external: bool = True,
    ) -> None:
        self.qval = qval
        self.test_func = test_func or _ident
        self.external = external

    def __call__(self, s1: Sequence, s2: Sequence) -> int:
        first = _prepare(s1, self.qval)
        second = _prepare(s2, self.qval)
        if _ident(first, second):
            return 0
        if not first or not second:
            return max(len(first), len(second))
        if self.test_func is not _ident:
            row = list(range(len(second) + 1))
            for i, left_value in enumerate(first, 1):
                diagonal = row[0]
                row[0] = i
                for j, right_value in enumerate(second, 1):
                    above = row[j]
                    row[j] = min(
                        above + 1,
                        row[j - 1] + 1,
                        diagonal + int(not self.test_func(left_value, right_value)),
                    )
                    diagonal = above
            return row[-1]
        left, right, _ = _dense_pair(first, second)
        if len(right) > len(left):
            left, right = right, left
        row = np.empty(len(right) + 1, dtype=np.int64)
        return int(
            lib().mtd_levenshtein(
                address(left), len(left), address(right), len(right), address(row, writable=True)
            )
        )


class DamerauLevenshtein(Base):
    def __init__(
        self,
        qval: int = 1,
        test_func: TestFunc | None = None,
        external: bool = True,
        restricted: bool = True,
    ) -> None:
        self.qval = qval
        self.test_func = test_func or _ident
        self.external = external
        self.restricted = restricted

    def __call__(self, s1: Sequence, s2: Sequence) -> int:
        first = _prepare(s1, self.qval)
        second = _prepare(s2, self.qval)
        if _ident(first, second):
            return 0
        if not first or not second:
            return max(len(first), len(second))
        if self.test_func is not _ident:
            return self._python(first, second)
        left, right, vocab = _dense_pair(first, second, exact_vocab=not self.restricted)
        if self.restricted:
            matrix = np.empty((len(left) + 1) * (len(right) + 1), dtype=np.int64)
            return int(
                lib().mtd_damerau_restricted(
                    address(left),
                    len(left),
                    address(right),
                    len(right),
                    address(matrix, writable=True),
                )
            )
        matrix = np.empty((len(left) + 2) * (len(right) + 2), dtype=np.int64)
        last = np.empty(max(vocab, 1), dtype=np.int64)
        return int(
            lib().mtd_damerau_unrestricted(
                address(left),
                len(left),
                address(right),
                len(right),
                address(matrix, writable=True),
                address(last, writable=True),
                vocab,
            )
        )

    def _python(self, first: Sequence, second: Sequence) -> int:
        if not self.restricted:
            distances: dict[tuple[int, int], int] = {}
            last: dict[Any, int] = {}
            maximum = len(first) + len(second)
            distances[-1, -1] = maximum
            for i in range(len(first) + 1):
                distances[i, -1] = maximum
                distances[i, 0] = i
            for j in range(len(second) + 1):
                distances[-1, j] = maximum
                distances[0, j] = j
            for i, left_value in enumerate(first, 1):
                match_column = 0
                for j, right_value in enumerate(second, 1):
                    match_row = last.get(right_value, 0)
                    previous_match_column = match_column
                    if self.test_func(left_value, right_value):
                        cost = 0
                        match_column = j
                    else:
                        cost = 1
                    distances[i, j] = min(
                        distances[i - 1, j - 1] + cost,
                        distances[i, j - 1] + 1,
                        distances[i - 1, j] + 1,
                        distances[match_row - 1, previous_match_column - 1]
                        + i
                        - match_row
                        - 1
                        + j
                        - previous_match_column,
                    )
                last[left_value] = i
            return distances[len(first), len(second)]
        rows, cols = len(first) + 1, len(second) + 1
        matrix = [[0] * cols for _ in range(rows)]
        for i in range(rows):
            matrix[i][0] = i
        for j in range(cols):
            matrix[0][j] = j
        for i in range(1, rows):
            for j in range(1, cols):
                cost = int(not self.test_func(first[i - 1], second[j - 1]))
                value = min(
                    matrix[i - 1][j] + 1,
                    matrix[i][j - 1] + 1,
                    matrix[i - 1][j - 1] + cost,
                )
                if (
                    i > 1
                    and j > 1
                    and self.test_func(first[i - 1], second[j - 2])
                    and self.test_func(first[i - 2], second[j - 1])
                ):
                    value = min(value, matrix[i - 2][j - 2] + cost)
                matrix[i][j] = value
        return matrix[-1][-1]


class JaroWinkler(BaseSimilarity):
    def __init__(
        self,
        long_tolerance: bool = False,
        winklerize: bool = True,
        qval: int = 1,
        external: bool = True,
    ) -> None:
        self.qval = qval
        self.long_tolerance = long_tolerance
        self.winklerize = winklerize
        self.external = external

    def maximum(self, *sequences: Sequence) -> int:
        return 1

    def __call__(
        self, s1: Sequence, s2: Sequence, prefix_weight: float = 0.1
    ) -> float:
        first = _prepare(s1, self.qval)
        second = _prepare(s2, self.qval)
        if _ident(first, second):
            return 1
        if not first or not second:
            return 0.0
        if max(len(first), len(second)) < 32:
            return _jaro_python(
                first,
                second,
                self.winklerize,
                self.long_tolerance,
                prefix_weight,
            )
        left, right, _ = _dense_pair(first, second)
        left_flags = np.empty(len(left), dtype=np.int64)
        right_flags = np.empty(len(right), dtype=np.int64)
        return float(
            lib().mtd_jaro(
                address(left),
                len(left),
                address(right),
                len(right),
                address(left_flags, writable=True),
                address(right_flags, writable=True),
                int(self.winklerize),
                int(self.long_tolerance),
                prefix_weight,
            )
        )


class Jaro(JaroWinkler):
    def __init__(
        self, long_tolerance: bool = False, qval: int = 1, external: bool = True
    ) -> None:
        super().__init__(long_tolerance, False, qval, external)


def _alignment_python(
    first: Sequence,
    second: Sequence,
    gap_cost: float,
    sim_func: SimFunc,
    local: bool,
) -> float:
    row = [-j * gap_cost for j in range(len(second) + 1)]
    for i, left_value in enumerate(first, 1):
        diagonal = row[0]
        row[0] = 0.0 if local else -i * gap_cost
        for j, right_value in enumerate(second, 1):
            above = row[j]
            value = max(
                diagonal + sim_func(left_value, right_value),
                above - gap_cost,
                row[j - 1] - gap_cost,
            )
            row[j] = max(0.0, value) if local else value
            diagonal = above
    return row[-1]


class NeedlemanWunsch(BaseSimilarity):
    def __init__(
        self,
        gap_cost: float = 1.0,
        sim_func: SimFunc | None = None,
        qval: int = 1,
        external: bool = True,
    ) -> None:
        self.qval = qval
        self.gap_cost = gap_cost
        self.sim_func = sim_func or _ident
        self.external = external

    def minimum(self, *sequences: Sequence) -> float:
        return -max(map(len, sequences)) * self.gap_cost

    def maximum(self, *sequences: Sequence) -> float:
        return max(map(len, sequences))

    def distance(self, *sequences: Sequence) -> float:
        return -self.similarity(*sequences)

    def normalized_distance(self, *sequences: Sequence) -> float:
        minimum = self.minimum(*sequences)
        maximum = self.maximum(*sequences)
        return 0 if maximum == 0 else (self.distance(*sequences) - minimum) / (
            maximum - minimum
        )

    def normalized_similarity(self, *sequences: Sequence) -> float:
        minimum = self.minimum(*sequences)
        maximum = self.maximum(*sequences)
        return 1 if maximum == 0 else (self.similarity(*sequences) - minimum) / (
            maximum * 2
        )

    def __call__(self, s1: Sequence, s2: Sequence) -> float:
        first, second, left, right, _ = _pair(s1, s2, self.qval)
        if self.sim_func is not _ident:
            return _alignment_python(
                first, second, self.gap_cost, self.sim_func, False
            )
        row = np.empty(len(right) + 1, dtype=np.float64)
        return float(
            lib().mtd_needleman_wunsch(
                address(left),
                len(left),
                address(right),
                len(right),
                self.gap_cost,
                address(row, writable=True),
            )
        )


class SmithWaterman(BaseSimilarity):
    def __init__(
        self,
        gap_cost: float = 1.0,
        sim_func: SimFunc | None = None,
        qval: int = 1,
        external: bool = True,
    ) -> None:
        self.qval = qval
        self.gap_cost = gap_cost
        self.sim_func = sim_func or _ident
        self.external = external

    def maximum(self, *sequences: Sequence) -> int:
        return min(map(len, sequences))

    def __call__(self, s1: Sequence, s2: Sequence) -> float:
        first, second, left, right, _ = _pair(s1, s2, self.qval)
        if _ident(first, second):
            return self.maximum(first, second)
        if not first or not second:
            return 0.0
        if self.sim_func is not _ident:
            return _alignment_python(first, second, self.gap_cost, self.sim_func, True)
        row = np.empty(len(right) + 1, dtype=np.float64)
        return float(
            lib().mtd_smith_waterman(
                address(left),
                len(left),
                address(right),
                len(right),
                self.gap_cost,
                address(row, writable=True),
            )
        )


def _assemble(values: list, original: Sequence, qval: int | None):
    if qval != 1:
        return values
    if isinstance(original, str):
        return "".join(values)
    if isinstance(original, bytes):
        return bytes(values)
    if isinstance(original, tuple):
        return tuple(values)
    return values


class LCSSeq(BaseSimilarity):
    def __init__(
        self,
        qval: int = 1,
        test_func: TestFunc | None = None,
        external: bool = True,
    ) -> None:
        self.qval = qval
        self.test_func = test_func or _ident
        self.external = external

    def __call__(self, *sequences: Sequence):
        if not sequences:
            return ""
        if len(sequences) == 1:
            return sequences[0]
        if len(sequences) != 2:
            result = sequences[0]
            for sequence in sequences[1:]:
                result = self(result, sequence)
            return result
        first = _prepare(sequences[0], self.qval)
        second = _prepare(sequences[1], self.qval)
        if not first or not second:
            return _assemble([], sequences[0], self.qval)
        left, right, _ = _dense_pair(first, second)
        rows, cols = len(left) + 1, len(right) + 1
        matrix = np.empty(rows * cols, dtype=np.int64)
        lib().mtd_lcsseq(
            address(left), len(left), address(right), len(right), address(matrix, writable=True)
        )
        i, j = len(left), len(right)
        values = []
        while i and j:
            current = matrix[i * cols + j]
            if current == matrix[(i - 1) * cols + j]:
                i -= 1
            elif current == matrix[i * cols + j - 1]:
                j -= 1
            else:
                values.append(first[i - 1])
                i -= 1
                j -= 1
        values.reverse()
        return _assemble(values, sequences[0], self.qval)

    def similarity(self, *sequences: Sequence) -> int:
        return len(self(*sequences))


class LCSStr(BaseSimilarity):
    def __call__(self, *sequences: Sequence):
        if not sequences:
            return ""
        if len(sequences) == 1:
            return sequences[0]
        if not all(sequences):
            return type(sequences[0])()
        prepared = [_prepare(sequence, self.qval) for sequence in sequences]
        if len(prepared) > 2:
            short = min(prepared, key=len)
            for size in range(len(short), 0, -1):
                for start in range(len(short) - size + 1):
                    candidate = short[start : start + size]
                    if all(
                        any(
                            sequence[index : index + size] == candidate
                            for index in range(len(sequence) - size + 1)
                        )
                        for sequence in prepared
                    ):
                        return _assemble(list(candidate), sequences[0], self.qval)
            return type(sequences[0])()
        if (
            max(map(len, prepared)) < 32
            and self.qval == 1
            and all(isinstance(sequence, (str, bytes)) for sequence in prepared)
        ):
            match = SequenceMatcher(a=prepared[0], b=prepared[1]).find_longest_match()
            return sequences[0][match.a : match.a + match.size]
        first_index = 0 if len(prepared[0]) <= len(prepared[1]) else 1
        first = prepared[first_index]
        second = prepared[1 - first_index]
        left, right, _ = _dense_pair(first, second)
        row = np.empty(len(right) + 1, dtype=np.int64)
        result = np.empty(2, dtype=np.int64)
        lib().mtd_lcsstr(
            address(left),
            len(left),
            address(right),
            len(right),
            address(row, writable=True),
            address(result, writable=True),
        )
        start, size = map(int, result)
        return _assemble(
            list(first[start : start + size]), sequences[first_index], self.qval
        )

    def similarity(self, *sequences: Sequence) -> int:
        return len(self(*sequences))


class RatcliffObershelp(BaseSimilarity):
    def maximum(self, *sequences: Sequence) -> int:
        return 1

    def _find(self, first: Sequence, second: Sequence) -> int:
        common = LCSStr(qval=1, external=False)(first, second)
        size = len(common)
        if not size:
            return 0
        left_start = first.find(common)
        right_start = second.find(common)
        return (
            self._find(first[:left_start], second[:right_start])
            + size
            + self._find(
                first[left_start + size :], second[right_start + size :]
            )
        )

    def __call__(self, *sequences: Sequence) -> float:
        if len(sequences) < 2:
            return 1
        if _ident(*sequences):
            return 1
        if not all(sequences):
            return 0
        if len(sequences) != 2:
            raise NotImplementedError("RatcliffObershelp currently accepts two sequences")
        return 2 * self._find(sequences[0], sequences[1]) / sum(map(len, sequences))


class Prefix(BaseSimilarity):
    def __init__(self, qval: int = 1, sim_test: SimFunc | None = None) -> None:
        self.qval = qval
        self.sim_test = sim_test or _ident

    def __call__(self, *sequences: Sequence):
        if not sequences:
            return ""
        prepared = [_prepare(sequence, self.qval) for sequence in sequences]
        if len(prepared) == 1:
            length = len(prepared[0])
        elif len(prepared) == 2 and self.sim_test is _ident:
            limit = min(map(len, prepared))
            if limit < 256:
                length = 0
                while (
                    length < limit and prepared[0][length] == prepared[1][length]
                ):
                    length += 1
            else:
                left, right, _ = _dense_pair(prepared[0], prepared[1])
                length = int(
                    lib().mtd_prefix(address(left), len(left), address(right), len(right))
                )
        else:
            length = 0
            for values in zip(*prepared):
                if not self.sim_test(*values):
                    break
                length += 1
        return _assemble(list(prepared[0][:length]), sequences[0], self.qval)

    def similarity(self, *sequences: Sequence) -> int:
        return len(self(*sequences))


class Postfix(Prefix):
    def __call__(self, *sequences: Sequence):
        original = sequences[0]
        prepared = [_prepare(sequence, self.qval) for sequence in sequences]
        if len(prepared) == 1:
            length = len(prepared[0])
        elif len(prepared) == 2 and self.sim_test is _ident:
            limit = min(map(len, prepared))
            if limit < 256:
                length = 0
                while (
                    length < limit
                    and prepared[0][-length - 1] == prepared[1][-length - 1]
                ):
                    length += 1
            else:
                left, right, _ = _dense_pair(prepared[0], prepared[1])
                length = int(
                    lib().mtd_postfix(address(left), len(left), address(right), len(right))
                )
        else:
            length = 0
            for values in zip(*(reversed(sequence) for sequence in prepared)):
                if not self.sim_test(*values):
                    break
                length += 1
        values = list(prepared[0][len(prepared[0]) - length :]) if length else []
        return _assemble(values, original, self.qval)


class Length(Base):
    def __call__(self, *sequences: Sequence) -> int:
        lengths = list(map(len, sequences))
        return max(lengths) - min(lengths)


class Identity(BaseSimilarity):
    def maximum(self, *sequences: Sequence) -> int:
        return 1

    def __call__(self, *sequences: Sequence) -> int:
        return int(_ident(*sequences))


class Matrix(BaseSimilarity):
    def __init__(
        self,
        mat=None,
        mismatch_cost: int = 0,
        match_cost: int = 1,
        symmetric: bool = True,
        external: bool = True,
    ) -> None:
        self.mat = mat
        self.mismatch_cost = mismatch_cost
        self.match_cost = match_cost
        self.symmetric = symmetric
        self.external = external

    def maximum(self, *sequences: Sequence) -> int:
        return self.match_cost

    def __call__(self, *sequences: Sequence) -> int:
        if self.mat:
            if sequences in self.mat:
                return self.mat[sequences]
            if self.symmetric and tuple(reversed(sequences)) in self.mat:
                return self.mat[tuple(reversed(sequences))]
        return self.match_cost if _ident(*sequences) else self.mismatch_cost


def _counter_stats(
    first: Sequence, second: Sequence, qval: int | None, as_set: bool
) -> tuple[int, int, int, int, int]:
    if isinstance(first, Counter) or isinstance(second, Counter):
        counters = [first if isinstance(first, Counter) else Counter(_prepare(first, qval))]
        counters.append(
            second if isinstance(second, Counter) else Counter(_prepare(second, qval))
        )
        left, right = counters
        if as_set:
            left = Counter({key: 1 for key in left})
            right = Counter({key: 1 for key in right})
        intersection = sum((left & right).values())
        union = sum((left | right).values())
        only_left = sum((left - right).values())
        only_right = sum((right - left).values())
        return intersection, union, sum(left.values()), sum(right.values()), max(
            only_left, only_right
        )
    prepared_first = _prepare(first, qval)
    prepared_second = _prepare(second, qval)
    if not prepared_first or not prepared_second:
        total_first = len(set(prepared_first)) if as_set else len(prepared_first)
        total_second = len(set(prepared_second)) if as_set else len(prepared_second)
        return (
            0,
            total_first + total_second,
            total_first,
            total_second,
            max(total_first, total_second),
        )
    left, right, vocab = _dense_pair(
        prepared_first, prepared_second, exact_vocab=True
    )
    counts_left = np.empty(vocab, dtype=np.int64)
    counts_right = np.empty(vocab, dtype=np.int64)
    stats = np.empty(5, dtype=np.int64)
    lib().mtd_token_stats(
        address(left),
        len(left),
        address(right),
        len(right),
        address(counts_left, writable=True),
        address(counts_right, writable=True),
        vocab,
        int(as_set),
        address(stats, writable=True),
    )
    return tuple(map(int, stats))


class _TokenSimilarity(BaseSimilarity):
    def __init__(
        self, qval: int = 1, as_set: bool = False, external: bool = True
    ) -> None:
        self.qval = qval
        self.as_set = as_set
        self.external = external

    def maximum(self, *sequences: Sequence) -> int:
        return 1

    def _quick(self, sequences: tuple[Sequence, ...]) -> float | None:
        if len(sequences) < 2 or _ident(*sequences):
            return 1
        if not all(sequences):
            return 0
        return None

    def _counts(self, sequences: tuple[Sequence, ...]) -> tuple[int, list[int]]:
        counters = [
            sequence
            if isinstance(sequence, Counter)
            else Counter(_prepare(sequence, self.qval))
            for sequence in sequences
        ]
        if self.as_set:
            counters = [Counter({key: 1 for key in counter}) for counter in counters]
        intersection = counters[0].copy()
        for counter in counters[1:]:
            intersection &= counter
        return sum(intersection.values()), [sum(counter.values()) for counter in counters]


class Jaccard(_TokenSimilarity):
    def __call__(self, *sequences: Sequence) -> float:
        quick = self._quick(sequences)
        if quick is not None:
            return quick
        if len(sequences) == 2:
            if (
                not isinstance(sequences[0], Counter)
                and not isinstance(sequences[1], Counter)
                and len(sequences[0]) + len(sequences[1]) < 256
            ):
                left = Counter(_prepare(sequences[0], self.qval))
                right = Counter(_prepare(sequences[1], self.qval))
                if self.as_set:
                    left = Counter({key: 1 for key in left})
                    right = Counter({key: 1 for key in right})
                return sum((left & right).values()) / sum(
                    (left | right).values()
                )
            intersection, union, _, _, _ = _counter_stats(
                sequences[0], sequences[1], self.qval, self.as_set
            )
        else:
            intersection, counts = self._counts(sequences)
            counters = [
                sequence
                if isinstance(sequence, Counter)
                else Counter(_prepare(sequence, self.qval))
                for sequence in sequences
            ]
            if self.as_set:
                counters = [Counter({key: 1 for key in counter}) for counter in counters]
            united = counters[0].copy()
            for counter in counters[1:]:
                united |= counter
            union = sum(united.values())
        return intersection / union


class Sorensen(_TokenSimilarity):
    def __call__(self, *sequences: Sequence) -> float:
        quick = self._quick(sequences)
        if quick is not None:
            return quick
        if len(sequences) == 2:
            intersection, _, first, second, _ = _counter_stats(
                sequences[0], sequences[1], self.qval, self.as_set
            )
            counts = [first, second]
        else:
            intersection, counts = self._counts(sequences)
        return 2.0 * intersection / sum(counts)


class Tversky(_TokenSimilarity):
    def __init__(
        self,
        qval: int = 1,
        ks: Sequence[float] | None = None,
        bias: float | None = None,
        as_set: bool = False,
        external: bool = True,
    ) -> None:
        super().__init__(qval, as_set, external)
        self.ks = ks or repeat(1)
        self.bias = bias

    def __call__(self, *sequences: Sequence) -> float:
        quick = self._quick(sequences)
        if quick is not None:
            return quick
        if len(sequences) == 2:
            intersection, _, first, second, _ = _counter_stats(
                sequences[0], sequences[1], self.qval, self.as_set
            )
            counts = [first, second]
        else:
            intersection, counts = self._counts(sequences)
        constants = list(islice(self.ks, len(counts)))
        if len(counts) != 2 or self.bias is None:
            denominator = intersection + sum(
                constant * (count - intersection)
                for constant, count in zip(constants, counts)
            )
            return intersection / denominator
        first, second = counts
        alpha, beta = constants
        a_value, b_value = min(first, second), max(first, second)
        c_value = intersection + self.bias
        denominator = alpha * beta * (a_value - b_value) + b_value * beta
        return c_value / (denominator + c_value)


class Overlap(_TokenSimilarity):
    def __call__(self, *sequences: Sequence) -> float:
        quick = self._quick(sequences)
        if quick is not None:
            return quick
        if len(sequences) == 2:
            intersection, _, first, second, _ = _counter_stats(
                sequences[0], sequences[1], self.qval, self.as_set
            )
            counts = [first, second]
        else:
            intersection, counts = self._counts(sequences)
        return intersection / min(counts)


class Cosine(_TokenSimilarity):
    def __call__(self, *sequences: Sequence) -> float:
        quick = self._quick(sequences)
        if quick is not None:
            return quick
        if len(sequences) == 2:
            intersection, _, first, second, _ = _counter_stats(
                sequences[0], sequences[1], self.qval, self.as_set
            )
            counts = [first, second]
        else:
            intersection, counts = self._counts(sequences)
        return intersection / pow(prod(counts), 1.0 / len(counts))


class Tanimoto(Jaccard):
    def __call__(self, *sequences: Sequence) -> float:
        similarity = super().__call__(*sequences)
        return float("-inf") if similarity == 0 else log(similarity, 2)


class Bag(Base):
    def __call__(self, *sequences: Sequence) -> float:
        if len(sequences) == 2:
            return _counter_stats(
                sequences[0], sequences[1], self.qval, False
            )[4]
        counters = [Counter(_prepare(sequence, self.qval)) for sequence in sequences]
        intersection = counters[0].copy()
        for counter in counters[1:]:
            intersection &= counter
        return max(sum((counter - intersection).values()) for counter in counters)
