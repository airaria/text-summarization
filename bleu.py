import math
from collections import Counter

def bleu(candidate, references, weights=[0.25]*4):
    p_ns = [
        _modified_precision(candidate, references, i)
        for i, _ in enumerate(weights, start=1)
    ]

    try:
        s = math.fsum(w * math.log(p_n) for w, p_n in zip(weights, p_ns))
    except ValueError:
        # some p_ns is 0
        return 0

    bp = _brevity_penalty(candidate, references)
    return bp * math.exp(s)


def _brevity_penalty(candidate, references):
    c = len(candidate)
    ref_lens = (len(reference) for reference in references)
    r = min(ref_lens, key=lambda ref_len: (abs(ref_len - c), ref_len))

    if c > r:
        return 1
    else:
        return math.exp(1 - r / c)

def _modified_precision(candidate, references, n):
    counts = Counter(_ngrams(candidate, n))

    if not counts:
        return 0

    max_counts = {}
    for reference in references:
        reference_counts = Counter(_ngrams(reference, n))
        for ngram in counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0), reference_counts[ngram])

    clipped_counts = dict((ngram, min(count, max_counts[ngram])) for ngram, count in counts.items())

    return sum(clipped_counts.values()) / sum(counts.values())

def _ngrams(sentence, n):
    if n==1:
        return sentence
    else:
        return [tuple(sentence[i:i+n]) for i in range(0,len(sentence)+1-n)]