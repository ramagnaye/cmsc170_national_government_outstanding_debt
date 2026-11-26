# Debt Ahead

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-arrays%20%26%20linalg-013243)
![Matplotlib](https://img.shields.io/badge/Matplotlib-visualizations-11557c)

Predicting whether the Philippine National Government’s outstanding debt will rise next month using historical debt trends. Implemented from scratch in Python using NumPy and Matplotlib, this project compares K-Nearest Neighbors, Naive Bayes, and the Perceptron against two baselines on monthly debt records from January 1993 to September 2025.

**Project Page:** [ramagnaye.github.io/cmsc170_national_government_outstanding_debt](https://ramagnaye.github.io/cmsc170_national_government_outstanding_debt/)

## Features

### Pipeline Capabilities

- **From-Scratch Data Loader** - Parses the 1993–2025 CSV, checks chronological month continuity and debt totals, computes month-over-month percentage changes, and standardizes features without `pandas` or `scikit-learn`.
- **From-Scratch Classifiers** - Implements K-Nearest Neighbors, Gaussian-Categorical Naive Bayes, and a Pocket Perceptron using only Python and NumPy.
- **Custom Evaluation Metrics** - Computes confusion matrix counts, accuracy, per-class precision/recall/F1, and Macro-F1 from scratch to handle class imbalance fairly.
- **Automated Comparison & Plots** - Evaluates all models and baselines on the same chronological 80/20 split, saving the comparison table (`comparison.txt`) and grouped bar chart (`scores.png`).

## Repository Structure

```text
cmsc170_national_government_outstanding_debt/
├── data/
│   └── debt_monthly_1993-2025.csv   # Monthly domestic, external, and total debt records
├── methods/
│   ├── knn.py                       # K-Nearest Neighbors classifier & validation k-selector
│   ├── naive_bayes.py               # Gaussian + Laplace-smoothed Categorical Naive Bayes
│   └── perceptron.py                # Pocket Perceptron binary classifier
├── results/
│   ├── comparison.txt               # Exported test-set metrics and confusion matrix table
│   └── scores.png                   # Grouped bar chart comparing method performance
├── data_loader.py                   # CSV parser, integrity checker, feature builder, and scaler
├── main.py                          # Experiment runner, table printer, and figure generator
├── metrics.py                       # From-scratch evaluation metrics (Accuracy, Per-Class F1, Macro-F1)
└── README.md                        # Project documentation, reproduction guide, and task log
```

## Prerequisites & Installation

### Requirements

- **Python 3.10+** (standard library: `csv`, `os`)
- **NumPy** (arrays and basic linear algebra only)
- **Matplotlib** (bar chart visualization)

*(High-level data and machine learning libraries such as `pandas`, `scikit-learn`, `PyTorch`, and `TensorFlow` are not used.)*

### Setup

1. Clone the repository and move into the project root:

```bash
git clone https://github.com/ramagnaye/cmsc170_national_government_outstanding_debt.git
cd cmsc170_national_government_outstanding_debt
```

2. Install the required packages:

```bash
pip install numpy matplotlib
```

## Reproducing the Results

All experiments are deterministic and reproducible from a single command. The pipeline splits the 391 labeled samples chronologically (approximately 80/20 without shuffling) into the first `313` months for training (`Feb 1993 – Feb 2019`) and the last `78` months for testing (`Mar 2019 – Aug 2025`), selects `k = 21` for KNN based on accuracy on the last 20% of the training set, and runs the Perceptron with a learning rate of `0.1`, up to `200` epochs, and `seed = 0`.

1. Run `main.py` from the **repository root**:

```bash
python main.py
```

*(On Windows, you can also run `py main.py`.)*

2. Running `main.py` automatically:
   - Trains KNN, Naive Bayes, and the Perceptron alongside the `Always up` and `Last-month trend` baselines on the same split.
   - Prints the evaluation table to the terminal.
   - Saves the table to `results/comparison.txt`.
   - Saves the grouped bar chart to `results/scores.png` and opens the plot window.

### Expected Output

Your terminal printout and `results/comparison.txt` should match these exact values:

```text
Method                 Acc  MacroF1  F1(up)  F1(not)   TN   FP   FN   TP
------------------------------------------------------------------------
KNN (k=21)           0.769    0.484   0.868    0.100    1   17    1   59
Naive Bayes          0.577    0.509   0.692    0.327    8   10   23   37
Perceptron           0.769    0.435   0.870    0.000    0   18    0   60
Always up            0.769    0.435   0.870    0.000    0   18    0   60
Last-month trend     0.679    0.540   0.793    0.286    5   13   12   48
```

## Running Individual Modules

To verify the data preprocessing pipeline (dataset counts, class balance, and Z-score scaling) or run each classifier standalone:

```bash
python data_loader.py
python methods/knn.py
python methods/naive_bayes.py
python methods/perceptron.py
```

## Task log

| Member                  | Contributions |
| ----------------------- | ------------- |
| Legasto, John Christian | TODO          |
| Magnaye, Roi Aaron      | TODO          |
| Montablan, Samuel R.    | TODO          |

## References & Disclosures

- **Dataset Source:** Extracted from *Outstanding Debt – September 2025 (Current Year Monthly)* [PDF], available on [BetterGov Data](https://data.bettergov.ph/).
- **AI Tool Disclosure:** TODO
