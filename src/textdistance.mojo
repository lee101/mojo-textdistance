"""String-distance kernels exposed through a C ABI.

Sequence elements are dense unsigned 32-bit token IDs. The caller owns all
buffers and scratch memory.
"""

from std.sys.info import simd_width_of

comptime TokenPtr = UnsafePointer[UInt32, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime F64Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]


def _minimum3(a: Int, b: Int, c: Int) -> Int:
    var value = a if a < b else b
    return value if value < c else c


@export("mtd_hamming")
def mtd_hamming(
    a_addr: Int, n: Int, b_addr: Int, m: Int, truncate: Int
) abi("C") -> Int:
    if n == 0 or m == 0:
        return 0 if truncate != 0 else (n if n > m else m)
    var a = TokenPtr(unsafe_from_address=a_addr)
    var b = TokenPtr(unsafe_from_address=b_addr)
    var limit = n if n < m else m
    var distance = 0
    for i in range(limit):
        if a[i] != b[i]:
            distance += 1
    if truncate == 0:
        distance += abs(n - m)
    return distance


@export("mtd_levenshtein")
def mtd_levenshtein(
    a_addr: Int, n: Int, b_addr: Int, m: Int, row_addr: Int
) abi("C") -> Int:
    if n == 0:
        return m
    if m == 0:
        return n
    var a = TokenPtr(unsafe_from_address=a_addr)
    var b = TokenPtr(unsafe_from_address=b_addr)
    var row = I64Ptr(unsafe_from_address=row_addr)
    for j in range(m + 1):
        row[j] = Int64(j)
    for i in range(1, n + 1):
        var diagonal = Int(row[0])
        row[0] = Int64(i)
        for j in range(1, m + 1):
            var above = Int(row[j])
            var substitution = diagonal + (0 if a[i - 1] == b[j - 1] else 1)
            row[j] = Int64(_minimum3(above + 1, Int(row[j - 1]) + 1, substitution))
            diagonal = above
    return Int(row[m])


@export("mtd_damerau_restricted")
def mtd_damerau_restricted(
    a_addr: Int, n: Int, b_addr: Int, m: Int, matrix_addr: Int
) abi("C") -> Int:
    if n == 0:
        return m
    if m == 0:
        return n
    var a = TokenPtr(unsafe_from_address=a_addr)
    var b = TokenPtr(unsafe_from_address=b_addr)
    var matrix = I64Ptr(unsafe_from_address=matrix_addr)
    var cols = m + 1
    for i in range(n + 1):
        matrix[i * cols] = Int64(i)
    for j in range(m + 1):
        matrix[j] = Int64(j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            var cost = 0 if a[i - 1] == b[j - 1] else 1
            var value = _minimum3(
                Int(matrix[(i - 1) * cols + j]) + 1,
                Int(matrix[i * cols + j - 1]) + 1,
                Int(matrix[(i - 1) * cols + j - 1]) + cost,
            )
            if (
                i > 1
                and j > 1
                and a[i - 1] == b[j - 2]
                and a[i - 2] == b[j - 1]
            ):
                var transposed = Int(matrix[(i - 2) * cols + j - 2]) + cost
                if transposed < value:
                    value = transposed
            matrix[i * cols + j] = Int64(value)
    return Int(matrix[n * cols + m])


@export("mtd_damerau_unrestricted")
def mtd_damerau_unrestricted(
    a_addr: Int,
    n: Int,
    b_addr: Int,
    m: Int,
    matrix_addr: Int,
    last_addr: Int,
    vocab: Int,
) abi("C") -> Int:
    if n == 0:
        return m
    if m == 0:
        return n
    var a = TokenPtr(unsafe_from_address=a_addr)
    var b = TokenPtr(unsafe_from_address=b_addr)
    var matrix = I64Ptr(unsafe_from_address=matrix_addr)
    var last = I64Ptr(unsafe_from_address=last_addr)
    var cols = m + 2
    var maximum = n + m
    for k in range(vocab):
        last[k] = 0
    matrix[0] = Int64(maximum)
    for i in range(n + 1):
        matrix[(i + 1) * cols] = Int64(maximum)
        matrix[(i + 1) * cols + 1] = Int64(i)
    for j in range(m + 1):
        matrix[j + 1] = Int64(maximum)
        matrix[cols + j + 1] = Int64(j)
    for i in range(1, n + 1):
        var match_column = 0
        for j in range(1, m + 1):
            var match_row = Int(last[Int(b[j - 1])])
            var previous_match_column = match_column
            var cost = 1
            if a[i - 1] == b[j - 1]:
                cost = 0
                match_column = j
            var value = _minimum3(
                Int(matrix[i * cols + j]) + cost,
                Int(matrix[(i + 1) * cols + j]) + 1,
                Int(matrix[i * cols + j + 1]) + 1,
            )
            var transposed = (
                Int(matrix[match_row * cols + previous_match_column])
                + (i - match_row - 1)
                + 1
                + (j - previous_match_column - 1)
            )
            if transposed < value:
                value = transposed
            matrix[(i + 1) * cols + j + 1] = Int64(value)
        last[Int(a[i - 1])] = Int64(i)
    return Int(matrix[(n + 1) * cols + m + 1])


@export("mtd_jaro")
def mtd_jaro(
    a_addr: Int,
    n: Int,
    b_addr: Int,
    m: Int,
    a_flags_addr: Int,
    b_flags_addr: Int,
    winklerize: Int,
    long_tolerance: Int,
    prefix_weight: Float64,
) abi("C") -> Float64:
    if n == 0 or m == 0:
        return 1.0 if n == m else 0.0
    var a = TokenPtr(unsafe_from_address=a_addr)
    var b = TokenPtr(unsafe_from_address=b_addr)
    var a_flags = I64Ptr(unsafe_from_address=a_flags_addr)
    var b_flags = I64Ptr(unsafe_from_address=b_flags_addr)
    for i in range(n):
        a_flags[i] = 0
    for j in range(m):
        b_flags[j] = 0
    var search_range = (n if n > m else m) // 2 - 1
    if search_range < 0:
        search_range = 0
    var common = 0
    for i in range(n):
        var low = i - search_range
        if low < 0:
            low = 0
        var high = i + search_range
        if high >= m:
            high = m - 1
        for j in range(low, high + 1):
            if b_flags[j] == 0 and a[i] == b[j]:
                a_flags[i] = 1
                b_flags[j] = 1
                common += 1
                break
    if common == 0:
        return 0.0
    var next_b = 0
    var transpositions = 0
    for i in range(n):
        if a_flags[i] != 0:
            while next_b < m and b_flags[next_b] == 0:
                next_b += 1
            if a[i] != b[next_b]:
                transpositions += 1
            next_b += 1
    transpositions //= 2
    var weight = (
        Float64(common) / Float64(n)
        + Float64(common) / Float64(m)
        + Float64(common - transpositions) / Float64(common)
    ) / 3.0
    if winklerize == 0 or weight <= 0.7:
        return weight
    var min_len = n if n < m else m
    var prefix_limit = min_len if min_len < 4 else 4
    var prefix = 0
    while prefix < prefix_limit and a[prefix] == b[prefix]:
        prefix += 1
    if prefix > 0:
        weight += Float64(prefix) * prefix_weight * (1.0 - weight)
    if long_tolerance == 0 or min_len <= 4:
        return weight
    if common <= prefix + 1 or 2 * common < min_len + prefix:
        return weight
    var adjustment = Float64(common - prefix - 1) / Float64(n + m - prefix * 2 + 2)
    return weight + (1.0 - weight) * adjustment


@export("mtd_lcsseq")
def mtd_lcsseq(
    a_addr: Int, n: Int, b_addr: Int, m: Int, matrix_addr: Int
) abi("C") -> Int:
    if n == 0 or m == 0:
        return 0
    var a = TokenPtr(unsafe_from_address=a_addr)
    var b = TokenPtr(unsafe_from_address=b_addr)
    var matrix = I64Ptr(unsafe_from_address=matrix_addr)
    var cols = m + 1
    for j in range(cols):
        matrix[j] = 0
    for i in range(1, n + 1):
        matrix[i * cols] = 0
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                matrix[i * cols + j] = matrix[(i - 1) * cols + j - 1] + 1
            else:
                var left = matrix[i * cols + j - 1]
                var above = matrix[(i - 1) * cols + j]
                matrix[i * cols + j] = left if left > above else above
    return Int(matrix[n * cols + m])


@export("mtd_lcsstr")
def mtd_lcsstr(
    a_addr: Int,
    n: Int,
    b_addr: Int,
    m: Int,
    row_addr: Int,
    result_addr: Int,
) abi("C") -> Int:
    if n == 0 or m == 0:
        return 0
    var a = TokenPtr(unsafe_from_address=a_addr)
    var b = TokenPtr(unsafe_from_address=b_addr)
    var row = I64Ptr(unsafe_from_address=row_addr)
    var result = I64Ptr(unsafe_from_address=result_addr)
    for j in range(m + 1):
        row[j] = 0
    var best = 0
    var best_start = 0
    for i in range(1, n + 1):
        for offset in range(m):
            var j = m - offset
            if a[i - 1] == b[j - 1]:
                row[j] = row[j - 1] + 1
                if Int(row[j]) > best:
                    best = Int(row[j])
                    best_start = i - best
            else:
                row[j] = 0
    result[0] = Int64(best_start)
    result[1] = Int64(best)
    return best


@export("mtd_prefix")
def mtd_prefix(a_addr: Int, n: Int, b_addr: Int, m: Int) abi("C") -> Int:
    if n == 0 or m == 0:
        return 0
    var a = TokenPtr(unsafe_from_address=a_addr)
    var b = TokenPtr(unsafe_from_address=b_addr)
    var limit = n if n < m else m
    var length = 0
    comptime W = simd_width_of[DType.uint32]()
    while length + W <= limit:
        if not a.load[width=W](length).eq(
            b.load[width=W](length)
        ).reduce_and():
            break
        length += W
    while length < limit and a[length] == b[length]:
        length += 1
    return length


@export("mtd_postfix")
def mtd_postfix(a_addr: Int, n: Int, b_addr: Int, m: Int) abi("C") -> Int:
    if n == 0 or m == 0:
        return 0
    var a = TokenPtr(unsafe_from_address=a_addr)
    var b = TokenPtr(unsafe_from_address=b_addr)
    var limit = n if n < m else m
    var length = 0
    comptime W = simd_width_of[DType.uint32]()
    while length + W <= limit:
        var a_start = n - length - W
        var b_start = m - length - W
        if not a.load[width=W](a_start).eq(
            b.load[width=W](b_start)
        ).reduce_and():
            break
        length += W
    while length < limit and a[n - length - 1] == b[m - length - 1]:
        length += 1
    return length


@export("mtd_token_stats")
def mtd_token_stats(
    a_addr: Int,
    n: Int,
    b_addr: Int,
    m: Int,
    counts_a_addr: Int,
    counts_b_addr: Int,
    vocab: Int,
    as_set: Int,
    stats_addr: Int,
) abi("C"):
    var a = TokenPtr(unsafe_from_address=a_addr)
    var b = TokenPtr(unsafe_from_address=b_addr)
    var counts_a = I64Ptr(unsafe_from_address=counts_a_addr)
    var counts_b = I64Ptr(unsafe_from_address=counts_b_addr)
    var stats = I64Ptr(unsafe_from_address=stats_addr)
    comptime W = simd_width_of[DType.float64]()
    var k = 0
    var zeros = SIMD[DType.int64, W](0)
    while k + W <= vocab:
        counts_a.store(k, zeros)
        counts_b.store(k, zeros)
        k += W
    while k < vocab:
        counts_a[k] = 0
        counts_b[k] = 0
        k += 1
    for i in range(n):
        counts_a[Int(a[i])] += 1
    for j in range(m):
        counts_b[Int(b[j])] += 1
    var intersection = 0
    var union_count = 0
    var total_a = 0
    var total_b = 0
    var only_a = 0
    var only_b = 0
    var ones = SIMD[DType.int64, W](1)
    k = 0
    while k + W <= vocab:
        var ca = counts_a.load[width=W](k)
        var cb = counts_b.load[width=W](k)
        if as_set != 0:
            ca = ca.gt(zeros).select(ones, zeros)
            cb = cb.gt(zeros).select(ones, zeros)
        intersection += Int(min(ca, cb).reduce_add())
        union_count += Int(max(ca, cb).reduce_add())
        total_a += Int(ca.reduce_add())
        total_b += Int(cb.reduce_add())
        only_a += Int(max(ca - cb, zeros).reduce_add())
        only_b += Int(max(cb - ca, zeros).reduce_add())
        k += W
    while k < vocab:
        var ca = Int(counts_a[k])
        var cb = Int(counts_b[k])
        if as_set != 0:
            ca = 1 if ca > 0 else 0
            cb = 1 if cb > 0 else 0
        intersection += ca if ca < cb else cb
        union_count += ca if ca > cb else cb
        total_a += ca
        total_b += cb
        if ca > cb:
            only_a += ca - cb
        else:
            only_b += cb - ca
        k += 1
    stats[0] = Int64(intersection)
    stats[1] = Int64(union_count)
    stats[2] = Int64(total_a)
    stats[3] = Int64(total_b)
    stats[4] = Int64(only_a if only_a > only_b else only_b)


@export("mtd_needleman_wunsch")
def mtd_needleman_wunsch(
    a_addr: Int, n: Int, b_addr: Int, m: Int, gap: Float64, row_addr: Int
) abi("C") -> Float64:
    if n == 0:
        return -Float64(m) * gap
    if m == 0:
        return -Float64(n) * gap
    var a = TokenPtr(unsafe_from_address=a_addr)
    var b = TokenPtr(unsafe_from_address=b_addr)
    var row = F64Ptr(unsafe_from_address=row_addr)
    for j in range(m + 1):
        row[j] = -Float64(j) * gap
    for i in range(1, n + 1):
        var diagonal = row[0]
        row[0] = -Float64(i) * gap
        for j in range(1, m + 1):
            var above = row[j]
            var match_score = diagonal + (1.0 if a[i - 1] == b[j - 1] else 0.0)
            var deletion = above - gap
            var insertion = row[j - 1] - gap
            var value = match_score if match_score > deletion else deletion
            row[j] = value if value > insertion else insertion
            diagonal = above
    return row[m]


@export("mtd_smith_waterman")
def mtd_smith_waterman(
    a_addr: Int, n: Int, b_addr: Int, m: Int, gap: Float64, row_addr: Int
) abi("C") -> Float64:
    if n == 0 or m == 0:
        return 0.0
    var a = TokenPtr(unsafe_from_address=a_addr)
    var b = TokenPtr(unsafe_from_address=b_addr)
    var row = F64Ptr(unsafe_from_address=row_addr)
    for j in range(m + 1):
        row[j] = 0.0
    for i in range(1, n + 1):
        var diagonal = row[0]
        row[0] = 0.0
        for j in range(1, m + 1):
            var above = row[j]
            var match_score = diagonal + (1.0 if a[i - 1] == b[j - 1] else 0.0)
            var deletion = above - gap
            var insertion = row[j - 1] - gap
            var value = match_score if match_score > deletion else deletion
            value = value if value > insertion else insertion
            row[j] = value if value > 0.0 else 0.0
            diagonal = above
    return row[m]
