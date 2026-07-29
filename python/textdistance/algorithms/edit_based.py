from ..core import (
    DamerauLevenshtein,
    Hamming,
    Jaro,
    JaroWinkler,
    Levenshtein,
    NeedlemanWunsch,
    SmithWaterman,
)

hamming = Hamming()
levenshtein = Levenshtein()
damerau_levenshtein = DamerauLevenshtein()
jaro = Jaro()
jaro_winkler = JaroWinkler()
needleman_wunsch = NeedlemanWunsch()
smith_waterman = SmithWaterman()

__all__ = [
    "Hamming",
    "Levenshtein",
    "DamerauLevenshtein",
    "Jaro",
    "JaroWinkler",
    "NeedlemanWunsch",
    "SmithWaterman",
    "hamming",
    "levenshtein",
    "damerau_levenshtein",
    "jaro",
    "jaro_winkler",
    "needleman_wunsch",
    "smith_waterman",
]
