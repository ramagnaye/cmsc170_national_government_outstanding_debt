# CMSC 170 - W3L
# Project: National Government Outstanding Debt

# usage in the other files:
#   from data_loader import prepare_data
#   data = prepare_data()

#   data["Xs_train"], data["Xs_test"]  scaled features (KNN, Perceptron)
#   data["X_train"], data["X_test"]    raw features (Naive Bayes)
#   data["y_train"], data["y_test"]    labels (1 = up next month, 0 = not up)

import csv
import os
import numpy as np

# all debt values are in million pesos
# the 4 features each month is described by
FEATURE_NAMES = ["dom_pct", "ext_pct", "tot_pct", "month_num"]

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "debt_monthly_1993-2025.csv",
)

# reads the csv and saves each month's debts as a dictionary
def load_debt_data(filename=DATASET_PATH):
    dataset = []
    # utf-8-sig ignores hidden markers some CSV files start with
    with open(filename, mode="r", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        for row in reader:
            # skip blank lines
            if not row:
                continue
            # skip header row
            if row[0].strip().lower() == "year":
                continue
 
            # save the month with its domestic, external, and total debt
            dataset.append(
                {
                    "year": int(row[0]),
                    "month": row[1].strip(),
                    "month_num": int(row[2]),
                    "dom": float(row[3]),
                    "ext": float(row[4]),
                    "tot": float(row[5]),
                }
            )
 
    return dataset

# checks that no month is missing and that domestic + external = total
# prints a warning and returns False if something is wrong
def check_data(dataset, tolerance=5.0):
    all_good = True
 
    for i in range(1, len(dataset)):
        # turn year and month into one month count so they can be compared
        prev_count = dataset[i - 1]["year"] * 12 + dataset[i - 1]["month_num"]
        cur_count = dataset[i]["year"] * 12 + dataset[i]["month_num"]
        if cur_count != prev_count + 1:
            month, year = dataset[i]["month"], dataset[i]["year"]
            print(f"Warning: gap or wrong order at {month} {year}")
            all_good = False
 
    for item in dataset:
        if abs(item["dom"] + item["ext"] - item["tot"]) > tolerance:
            month, year = item["month"], item["year"]
            print(f"Warning: dom + ext does not match total at {month} {year}")
            all_good = False
 
    return all_good

# makes the samples (X) and labels (y), one for each month
# X = % change of domestic, external, and total debt from last month, and the month number
# y = 1 if total debt goes up next month, 0 if not
# the first and last month are skipped since they have no last/next month
def build_dataset(dataset):
    X = []
    y = []
    tags = []
 
    for t in range(1, len(dataset) - 1):
        last = dataset[t - 1]
        now = dataset[t]
        nxt = dataset[t + 1]
 
        X.append(
            [
                (now["dom"] / last["dom"] - 1) * 100,
                (now["ext"] / last["ext"] - 1) * 100,
                (now["tot"] / last["tot"] - 1) * 100,
                now["month_num"],
            ]
        )
        y.append(1 if nxt["tot"] > now["tot"] else 0)
        tags.append((now["year"], now["month"]))
 
    return np.array(X, dtype=float), np.array(y, dtype=int), tags

# splits the samples into train and test WITHOUT shuffling
def split_data(X, y, test_ratio=0.2):
    cut = int(round(len(X) * (1 - test_ratio)))
    return X[:cut], X[cut:], y[:cut], y[cut:]

# gets the mean and standard deviation of each feature from the TRAIN set only
# KNN and Perceptron uses this to scale the features
def get_scaling(X_train):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    # avoid division by zero
    std[std == 0] = 1.0
    return mean, std

# z = (x - mean) / std, using the mean and std from the train set
def scale(X, mean, std):
    return (X - mean) / std

# runs everything above in one call so every method gets the same data
# returns a dictionary of the raw (X_) and scaled (Xs_) sets, labels, tags, and mean/std
def prepare_data(filename=DATASET_PATH, test_ratio=0.2):
    dataset = load_debt_data(filename)
    if not check_data(dataset):
        raise ValueError("data check failed, see the warnings above")
 
    X, y, tags = build_dataset(dataset)
    X_train, X_test, y_train, y_test = split_data(X, y, test_ratio)
 
    mean, std = get_scaling(X_train)
    cut = len(X_train)
 
    return {
        "X_train": X_train,
        "X_test": X_test,
        "Xs_train": scale(X_train, mean, std),
        "Xs_test": scale(X_test, mean, std),
        "y_train": y_train,
        "y_test": y_test,
        "tags_train": tags[:cut],
        "tags_test": tags[cut:],
        "mean": mean,
        "std": std,
        "feature_names": FEATURE_NAMES,
    }

# loads the data and prints a summary to show it loaded correctly
def main():
    data = prepare_data()
 
    y_train, y_test = data["y_train"], data["y_test"]
    tags_train, tags_test = data["tags_train"], data["tags_test"]
    y_all = np.concatenate([y_train, y_test])
 
    print(f"Samples: {len(y_all)}, features: {data['feature_names']}")
    print(f"Up: {int(y_all.sum())}, not up: {int(len(y_all) - y_all.sum())}")
    print(f"Always-predict-up baseline: {max(y_all.mean(), 1 - y_all.mean()):.3f}")
    print(
        f"Train: {len(y_train)} ({tags_train[0][1]} {tags_train[0][0]} - "
        f"{tags_train[-1][1]} {tags_train[-1][0]})"
    )
    print(
        f"Test: {len(y_test)} ({tags_test[0][1]} {tags_test[0][0]} - "
        f"{tags_test[-1][1]} {tags_test[-1][0]})"
    )
    print(f"Train up-rate: {y_train.mean():.3f}, test up-rate: {y_test.mean():.3f}")
    print("Scaled train means:", np.round(data["Xs_train"].mean(axis=0), 6))
 
 
if __name__ == "__main__":
    main()