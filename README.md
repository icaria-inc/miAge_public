# Urinary miRNA Age Prediction

This repository provides two Python scripts for age prediction as described by Havelka et al. (2025) in "A urinary microRNA aging clock accurately predicts biological age". The scripts implement a LightGBM regression model trained on urinary miRNA features, with nested cross-validation, hyperparameter tuning using Optuna, and bias correction.

* **Training**: `train_model.py` performs nested stratified cross-validation, Optuna hyperparameter tuning, bias correction, and generates performance metrics, feature importances, and diagnostic plots.
* **Validation**: `validate_model.py` loads the trained model and precomputed offset map to evaluate on a held-out dataset, applies bias correction by default, computes performance metrics, and produces an optional scatter plot.

## Repository Structure

```
├── train_model.py        # Training script
├── validate_model.py     # Validation script
└── README.md             # This file
```

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/icaria-inc/miAge_public.git
   cd miAge_public
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install pandas numpy scikit-learn scipy matplotlib lightgbm optuna joblib
   ```

## Data Preparation

* **Format**: Input CSV must have samples as rows, feature columns (e.g., miRNA counts), and include `age` and `gender` columns.
* **Sample filtering**: Only samples with total counts ≥100,000 were used in the original study. However, this can be adjusted based on a user-defined threshold. 
* **Low‐information feature filtering**:
  * Remove any feature that is zero or missing in all samples.
  * Require each feature to have at least `value_threshold` counts in more than `fraction_threshold` of samples (e.g., ≥1 count in >50% of samples).
* **CPM normalization**: Normalize counts to CPM (counts per million).

## `train_model.py`

### Description

Performs nested stratified cross-validation and Optuna-based hyperparameter tuning to train a LightGBM regression model for age prediction from miRNA features. Implements bias correction and generates performance metrics, feature importances, and diagnostic plots.

### Usage

```bash
python train_model.py \
    --input-file path/to/train.csv \
    --results-dir output/ \
    [--target-col age] \
    [--id-col sampleID] \
    [--gender-col gender] \
    [--n-splits 5] \
    [--inner-splits 5] \
    [--quantile-bins 5] \
    [--n-trials 100] \
    [--early-stopping 20] \
    [--metric rmse] \
    [--no-plot] \
    [--seed 42] \
    [--verbose] \
    [--help-extended]
```

### Arguments

* `--input-file` (str, **required**): Path to input CSV. Must contain columns: sample ID, target age, gender, and feature columns.
* `--results-dir` (str, **required**): Directory where outputs (models, CSVs, plots) are saved.
* `--target-col` (str): Name of the target column (default: `age`).
* `--id-col` (str): Name of the sample ID column (default: `sampleID`).
* `--gender-col` (str): Name of the gender column (default: `gender`).
* `--n-splits` (int): Number of outer stratified CV folds (default: 5).
* `--inner-splits` (int): Number of inner CV folds for tuning (default: 5).
* `--quantile-bins` (int): Number of quantile bins for stratification (default: 5).
* `--n-trials` (int): Number of Optuna trials (default: 100).
* `--early-stopping` (int): Early stopping rounds for LightGBM (default: 20).
* `--metric` (str): LightGBM eval metric, `rmse` or `mae` (default: `rmse`).
* `--no-plot` (flag): Disable generation of scatter plot.
* `--seed` (int): Random seed (default: 42).
* `--verbose` (flag): Enable debug logging.
* `--help-extended` (flag): Print full docstring and exit.

### Outputs

Inside `results-dir`, the script creates:

* `best_model.joblib`: Serialized model and offset map.
* `cv_metrics.csv`: Fold-wise CV metrics (MAE, MSE, R²).
* `feature_importances.csv`: Average feature importance across folds.
* `combined_results.csv`: Actual vs. predicted ages and gender for all samples.
* `bias_plot.png`: Scatter plot of predicted vs. actual age.

## `validate_model.py`

### Description

Loads a trained LightGBM model and its precomputed offset map to predict ages on a validation dataset. Applies bias correction by default, computes metrics, and optionally generates a scatter plot by gender.

### Usage

```bash
python validate_model.py \
    --model-path path/to/best_model.joblib \
    --validation-file path/to/validation.csv \
    --results-dir output/ \
    [--target-col age] \
    [--id-col sampleID] \
    [--gender-col gender] \
    [--no-plot] \
    [--seed 42] \
    [--verbose] \
    [--help-extended]
```

### Arguments

* `--model-path` (str, **required**): Path to `.joblib` file saved by `train_model.py`.
* `--validation-file` (str, **required**): Path to validation CSV file.
* `--results-dir` (str, **required**): Directory for metrics CSV and plot.
* `--target-col` (str): Name of the target age column (default: `age`).
* `--id-col` (str): Name of the sample ID column (default: `sampleID`).
* `--gender-col` (str): Name of the gender column (default: `gender`).
* `--no-plot` (flag): Disable scatter plot.
* `--seed` (int): Random seed (default: 42).
* `--verbose` (flag): Enable debug logging.
* `--help-extended` (flag): Print full docstring and exit.

### Outputs

Inside `results-dir`, the script creates:

* `validation_metrics.csv`: Combined MAE, MSE, R², and Pearson R.
* `validation_results.csv`: Actual vs. predicted ages and gender for each sample.
* `validation_scatter.png`: Diagnostic scatter plot by gender.

## License

This project is licensed for **non-commercial use only**. You are free to use, modify, and distribute this software for academic, research, and personal purposes, provided that no part of it is used for commercial gain.

For questions or special licensing arrangements, please contact yuki.ichikawa@craif.com.
