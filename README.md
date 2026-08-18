# Synthetic Fraud Dataset Validation

Can a synthetic copy of an insurance fraud dataset replace the real one — without losing the
fraud signal, and without exposing real claimants?

This project takes a real vehicle insurance claims dataset, generates a synthetic copy with
**CTGAN**, and puts that copy through a **five-level validation framework**. Everything lives in
one notebook: `synthetic_fraud_validation.ipynb`.

Coursework for DAT610 (Ethics and Privacy).

---

## The data

[Vehicle Insurance Fraud Detection](https://www.kaggle.com/datasets/khusheekapoor/vehicle-insurance-fraud-detection)
(Kaggle), included as `data/carclaims.csv`.

- 15,420 claims, 32 columns after dropping `PolicyNumber` (it is just the row number)
- 7 numeric columns, 25 categorical
- **5.99% fraud** — 923 fraudulent claims. This imbalance shapes every result below.

## The generator

CTGAN, 300 epochs, batch size 500, ~34 minutes on CPU. Synthetic numeric values are rounded to
whole numbers and clipped to the real min/max, since CTGAN returns things like `Year = 1994.83`.

## The five levels

| Level | Question it answers | Method |
|---|---|---|
| 1 | Same centre and spread? | Means, std, percentiles |
| 2 | Same shapes and relationships? | KDE plots, bar charts, TVD, correlation heatmaps |
| 3 | Are the differences detectable, and how big? | KS test, chi-squared, Wasserstein, Cramér's V |
| 4 | Does it still catch real fraud? | TSTR vs TRTR with a Random Forest |
| 5 | Did it memorise real people? | NNDR, DCR, DNNR against a real-vs-real baseline |


---

## Running it

```bash
./setup_env.sh                  # creates .venv, installs pinned requirements
source .venv/bin/activate
jupyter notebook synthetic_fraud_validation.ipynb
```

Then Run All. Expect ~40 minutes, most of it CTGAN training.

Reproducibility is pinned three ways: `SEED = 42`, `torch.set_num_threads(4)` (thread count
changes floating-point summation order, which compounds over 300 epochs), and exact versions in
`requirements.txt`. Loosening any pin may change the numbers above.

## Layout

```
data/carclaims.csv                      real dataset (input)
synthetic_fraud_validation.ipynb        the whole analysis
setup_env.sh, requirements.txt          environment
figures/                                8 plots, written by the notebook
outputs/
  real_carclaims_processed.csv          real data after preprocessing
  synthetic_carclaims_ctgan.csv         15,420 synthetic claims
  validation_metrics.json               every number in this README
```

`outputs/` and `figures/` are regenerated on every run.
