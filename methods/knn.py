# K-Nearest Neighbors

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import prepare_data

def train(Xs_train, y_train, k=5):
    return {"k": k, "X": Xs_train, "y": y_train}

def predict(model, Xs):
    # Euclidean distance from every test row to every training row, shape (n_test, n_train)
    dists = np.linalg.norm(Xs[:, None, :] - model["X"][None, :, :], axis=2)

    # Indices of the k closest training rows for each test row
    nearest = np.argsort(dists, axis=1)[:, :model["k"]]

    # Majority vote among the k neighbors' labels
    votes = model["y"][nearest].mean(axis=1)

    # Returns the predicted classification (If debt increases or decreases)
    return (votes >= 0.5).astype(int)

def choose_k(Xs_train, y_train, candidates=(1, 3, 5, 7, 11, 15, 21)):
    # Picks k using the LAST 20% of the training set as validation (no shuffling,
    # same idea as the main split), so the test set is never used for tuning
    cut = int(round(len(Xs_train) * 0.8))
    best_k, best_acc = candidates[0], -1.0

    for k in candidates:
        model = train(Xs_train[:cut], y_train[:cut], k)
        acc = np.mean(predict(model, Xs_train[cut:]) == y_train[cut:])
        if acc > best_acc:
            best_k, best_acc = k, acc

    return best_k


def main():
    data = prepare_data()

    k = choose_k(data["Xs_train"], data["y_train"])
    print(f"Chosen k: {k}")

    model = train(data["Xs_train"], data["y_train"], k)

    # From KNN
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