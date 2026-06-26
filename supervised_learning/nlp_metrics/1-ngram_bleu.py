#!/usr/bin/env python3
"""N-gram BLEU score module."""
from collections import Counter
from math import exp


def ngram_bleu(references, sentence, n):
    """Calculate the n-gram BLEU score for a sentence."""
    def get_ngrams(words, n):
        """Extract all n-grams from a list of words."""
        return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]

    sentence_ngrams = Counter(get_ngrams(sentence, n))
    clipped_count = 0

    for ngram, count in sentence_ngrams.items():
        max_count = max(
            get_ngrams(reference, n).count(ngram) for reference in references
        )
        clipped_count += min(count, max_count)

    total_ngrams = max(len(sentence) - n + 1, 0)
    if total_ngrams == 0:
        return 0

    precision = clipped_count / total_ngrams

    sentence_len = len(sentence)
    ref_lens = [len(reference) for reference in references]
    closest_ref_len = min(
        ref_lens,
        key=lambda ref_len: (abs(ref_len - sentence_len), ref_len)
    )

    if sentence_len > closest_ref_len:
        bp = 1
    else:
        bp = exp(1 - (closest_ref_len / sentence_len))

    return bp * precision
