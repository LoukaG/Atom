import numpy as np


def lexical_richness(texts):
    words = [w for t in texts for w in t.lower().split()]
    return len(set(words)) / len(words) if words else 0.0


def typo_rate(texts):
    ratios = []
    for t in texts:
        words = t.split()
        if words:
            ratios.append(sum(1 for w in words if not w.isascii()) / len(words))
    return float(np.mean(ratios)) if ratios else 0.0
