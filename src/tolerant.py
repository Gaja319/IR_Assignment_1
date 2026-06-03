from collections import defaultdict


def build_kgram_index(vocab, k=2):
    idx = defaultdict(set)
    for term in vocab:
        term2 = f"${term}$"
        for i in range(len(term2)-k+1):
            gram = term2[i:i+k]
            idx[gram].add(term)
    return idx


def candidate_terms_from_kgrams(kgram_index, pattern, k=2):
    pat = f"${pattern}$"
    grams = [pat[i:i+k] for i in range(len(pat)-k+1)]
    candidates = None
    for g in grams:
        s = kgram_index.get(g, set())
        if candidates is None:
            candidates = set(s)
        else:
            candidates &= s
    return candidates or set()


def edit_distance(a: str, b: str) -> int:
    m = len(a)
    n = len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1,m+1):
        for j in range(1,n+1):
            cost = 0 if a[i-1]==b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[m][n]


def spell_correction(term, vocab, kgram_index=None, max_edit=2):
    candidates = None
    if kgram_index is not None:
        candidates = candidate_terms_from_kgrams(kgram_index, term)
    else:
        candidates = vocab
    best = []
    for c in candidates:
        d = edit_distance(term, c)
        if d <= max_edit:
            best.append((c,d))
    best.sort(key=lambda x: x[1])
    return [b[0] for b in best]
