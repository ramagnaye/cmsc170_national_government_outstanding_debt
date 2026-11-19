# Perceptron

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import prepare_data


def train(Xs_train, y_train, lr=0.1, epochs=200, seed=0):
    rng = np.random.default_rng(seed)

    # Starts the weights and bias at zero
    w = np.zeros(Xs_train.shape[1])
    b = 0.0

    # Keeps the best weights found so far, since the data is probably not linearly separable
    best_w, best_b = w.copy(), b
    best_errors = len(y_train) + 1

    for _ in range(epochs):
        # Goes through the training rows in a random order every epoch
        for i in rng.permutation(len(Xs_train)):
            # Predicts 1 if the weighted sum of the features is above 0, else 0
            pred = 1 if Xs_train[i] @ w + b > 0 else 0

            # Updates the weights and bias only when the prediction is wrong
            w += lr * (y_train[i] - pred) * Xs_train[i]
            b += lr * (y_train[i] - pred)

        # Counts how many training rows are still misclassified
        errors = np.sum((Xs_train @ w + b > 0).astype(int) != y_train)

        # Saves the weights if they make fewer mistakes than the best so far
        if errors < best_errors:
            best_w, best_b, best_errors = w.copy(), b, errors

        # Stops early when every training row is classified correctly
        if errors == 0:
            break

    # Returns the best weights and bias
    return {"w": best_w, "b": best_b}


def predict(model, Xs):
    # Calculates the weighted sum of the features plus the bias for each row
    # Returns the predicted classification (If debt increases or decreases)
    return (Xs @ model["w"] + model["b"] > 0).astype(int)


def main():
    data = prepare_data()
    model = train(data["Xs_train"], data["y_train"])

    # From perceptron
    predictions = predict(model, data["Xs_test"])

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