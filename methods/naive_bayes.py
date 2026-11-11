# Naive Bayes

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import prepare_data


def train(X_train, y_train):
    model = {}

    for label in (0, 1):
        group = X_train[y_train == label]

        # Calculates probability on how common each class is before considering the feature
        prior = len(group) / len(X_train)

        # Calculates the average percentage change of each debt feature, calculates variance of each debt feature
        mean = group[:, :3].mean(axis=0)
        var = group[:, :3].var(axis=0) + 1e-9

        # Count January-December occurrences and applies Laplace smoothing.
        counts = np.bincount(group[:, 3].astype(int), minlength=13)[1:13]
        month_probs = (counts + 1) / (len(group) + 12)

        # Returns a dictionary with the list of values
        model[label] = (prior, mean, var, month_probs)

    return model


def predict(model, X):
    scores = []

    for label in (0, 1):
        prior, mean, var, month_probs = model[label]

        # Gaussian + Categorical Naive Bayes instead of Multinomial Naive Bayes
        gaussian = -0.5 * np.sum(np.log(2 * np.pi * var) + (X[:, :3] - mean) ** 2 / var, axis=1,)
        months = X[:, 3].astype(int)
        score = np.log(prior) + gaussian + np.log(month_probs[months - 1])
        scores.append(score)

    # Returns the predicted classification (If debt increases or decreases)
    return np.argmax(np.array(scores), axis=0)


def main():
    data = prepare_data()
    model = train(data["X_train"], data["y_train"])

    # From naive bayes
    predictions = predict(model, data["X_test"])

    # Actual correct classifications
    actual = data["y_test"]

    tp = np.sum((predictions == 1) & (actual == 1))
    tn = np.sum((predictions == 0) & (actual == 0))
    fp = np.sum((predictions == 1) & (actual == 0))
    fn = np.sum((predictions == 0) & (actual == 1))

    # Temporary evaluation metrics (since not discussed yet)
    accuracy = (tp + tn) / len(actual)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")


if __name__ == "__main__":
    main()
