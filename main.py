# Method Comparison

import os
import numpy as np
import matplotlib.pyplot as plt

from data_loader import prepare_data
from metrics import evaluate
from methods import knn, naive_bayes, perceptron

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def main():
    data = prepare_data()

    # Trains each method and gets predictions
    k = knn.choose_k(data["Xs_train"], data["y_train"])
    predictions = {
        f"KNN (k={k})": knn.predict(knn.train(data["Xs_train"], data["y_train"], k), data["Xs_test"]),
        "Naive Bayes": naive_bayes.predict(naive_bayes.train(data["X_train"], data["y_train"]), data["X_test"]),
        "Perceptron": perceptron.predict(perceptron.train(data["Xs_train"], data["y_train"], seed=0), data["Xs_test"]),
        "Always up": np.ones(len(data["y_test"]), dtype=int),
        "Last-month trend": (data["X_test"][:, 2] > 0).astype(int),
    }

    # Evaluates each method's predictions against the actual test labels
    results = {name: evaluate(data["y_test"], pred) for name, pred in predictions.items()}

    # Builds, prints, and saves the comparison table
    header = f"{'Method':<18}{'Acc':>8}{'MacroF1':>9}{'F1(up)':>8}{'F1(not)':>9}{'TN':>5}{'FP':>5}{'FN':>5}{'TP':>5}"
    lines = [header, "-" * len(header)]
    for name, r in results.items():
        lines.append(
            f"{name:<18}{r['accuracy']:>8.3f}{r['macro_f1']:>9.3f}"
            f"{r['up_f1']:>8.3f}{r['not_f1']:>9.3f}"
            f"{r['tn']:>5}{r['fp']:>5}{r['fn']:>5}{r['tp']:>5}"
        )
    table = "\n".join(lines)
    print(table)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "comparison.txt"), "w", encoding="utf-8") as f:
        f.write(table + "\n")

    # Bar chart of accuracy and F1 scores for each method
    bars = [
        ("Accuracy", "accuracy"),
        ("Macro F1", "macro_f1"),
        ("F1 (up)", "up_f1"),
        ("F1 (not up)", "not_f1"),
    ]

    x = np.arange(len(results))
    width = 0.8 / len(bars)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (label, key) in enumerate(bars):
        offset = (i - (len(bars) - 1) / 2) * width
        ax.bar(x + offset, [results[m][key] for m in results], width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(results.keys())
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Comparison of Methods on Test Set")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "scores.png"), dpi=150)
    plt.show()

if __name__ == "__main__":
    main()