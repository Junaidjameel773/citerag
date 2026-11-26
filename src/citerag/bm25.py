import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def bm25_scores(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Okapi BM25 score of each doc for the query; 0 means no query term occurs in it.

    Uses Lucene's IDF, log(1 + (N - df + 0.5) / (df + 0.5)), which stays positive, so a
    term found in half of a tiny corpus still counts (plain Okapi IDF would be 0 there)."""
    tokenized = [tokenize(d) for d in docs]
    if not tokenized:
        return []
    n = len(tokenized)
    avgdl = sum(map(len, tokenized)) / n or 1.0
    df = Counter(t for doc in tokenized for t in set(doc))
    terms = set(tokenize(query))
    scores = []
    for doc in tokenized:
        tf = Counter(doc)
        norm = k1 * (1 - b + b * len(doc) / avgdl)
        scores.append(
            sum(
                math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) * tf[t] * (k1 + 1) / (tf[t] + norm)
                for t in terms
                if tf[t]
            )
            + 0.0  # float even when nothing matches
        )
    return scores
