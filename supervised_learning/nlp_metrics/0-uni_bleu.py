#!/usr/bin/env python3
"""Unigram BLEU score module."""
from collections import Counter
from math import exp


def uni_bleu(references, sentence):
    """Calculate the unigram BLEU score for a sentence."""
    if len(sentence) == 0:
        return 0

    sentence_count = Counter(sentence)
    clipped_count = 0

    for word, count in sentence_count.items():
        max_count = max(reference.count(word) for reference in references)
        clipped_count += min(count, max_count)

    precision = clipped_count / len(sentence)

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
