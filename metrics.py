# Evaluation Metrics

import numpy as np

def confusion_counts(actual, predicted, positive=1):
    # Counts true positives, true negatives, false positives, and false negatives
    tp = int(np.sum((predicted == positive) & (actual == positive)))
    fp = int(np.sum((predicted == positive) & (actual != positive)))
    fn = int(np.sum((predicted != positive) & (actual == positive)))
    tn = int(np.sum((predicted != positive) & (actual != positive)))
    return tp, tn, fp, fn

def accuracy(tp, tn, fp, fn):
    # Calculates the share of total predictions that are correct
    return (tp + tn) / (tp + tn + fp + fn)

def precision(tp, fp):
    # Calculates how often positive predictions are actually correct
    return tp / (tp + fp) if tp + fp else 0.0

def recall(tp, fn):
    # Calculates how many actual positives were caught
    return tp / (tp + fn) if tp + fn else 0.0

def f1_score(p, r):
    # Combines precision and recall into one score
    return 2 * p * r / (p + r) if p + r else 0.0

def evaluate(actual, predicted):
    # Counts confusion matrix outcomes for both up (1) and not up (0)
    tp, tn, fp, fn = confusion_counts(actual, predicted, positive=1)
    tp0, tn0, fp0, fn0 = confusion_counts(actual, predicted, positive=0)

    # Calculates precision, recall, and F1 for each class
    up_p, up_r = precision(tp, fp), recall(tp, fn)
    not_p, not_r = precision(tp0, fp0), recall(tp0, fn0)
    up_f1 = f1_score(up_p, up_r)
    not_f1 = f1_score(not_p, not_r)

    # Returns all metrics
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy(tp, tn, fp, fn),
        "up_precision": up_p,
        "up_recall": up_r,
        "up_f1": up_f1,
        "not_precision": not_p,
        "not_recall": not_r,
        "not_f1": not_f1,
        "macro_f1": (up_f1 + not_f1) / 2,
    }