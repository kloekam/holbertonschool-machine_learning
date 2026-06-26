#!/usr/bin/env python3
"""Cumulative N-gram BLEU score module."""
from collections import Counter
from math import exp, log


def cumulative_bleu(references, sentence, n):
    """Calculate the cumulative n-gram BLEU score for a sentence."""
    def get_ngrams(words, k):
        """Extract all k-grams from a list of words."""
        return [tuple(words[i:i + k]) for i in range(len(words) - k + 1)]

    def ngram_precision(k):
        """Calculate clipped precision for k-grams."""
        sentence_ngrams = Counter(get_ngrams(sentence, k))
        clipped_count = 0
        for ngram, count in sentence_ngrams.items():
            max_count = max(
                get_ngrams(ref, k).count(ngram) for ref in references
            )
            clipped_count += min(count, max_count)
        total = max(len(sentence) - k + 1, 0)
        if total == 0:
            return 0
        return clipped_count / total

    weight = 1 / n
    log_sum = sum(weight * log(ngram_precision(k) or 1e-300)
                  for k in range(1, n + 1))

    sentence_len = len(sentence)
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(
        ref_lens,
        key=lambda ref_len: (abs(ref_len - sentence_len), ref_len)
    )

    if sentence_len > closest_ref_len:
        bp = 1
    else:
        bp = exp(1 - (closest_ref_len / sentence_len))

    return bp * exp(log_sum)
