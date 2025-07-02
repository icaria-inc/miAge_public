#!/usr/bin/env python3
"""
Validate an age prediction model using LightGBM and precomputed offset map.

This script loads a trained model and offset mapping, applies it to a
validation dataset, computes performance metrics, and generates a
scatter plot by gender. Bias correction is always applied.

Usage:
    python validate_model.py \
        --model-path path/to/best_model.joblib \
        --validation-file path/to/validation.csv \
        --results-dir results/ \
        [--target-col age] \
        [--id-col sampleID] \
        [--gender-col gender] \
        [--no-plot] \
        [--seed 42] \
        [--verbose] \
        [--help-extended]
"""
import argparse
import os
import sys
import logging
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate age prediction model using LightGBM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    req = parser.add_argument_group('required arguments')
    req.add_argument("--model-path", type=str, required=True,
                     help="Path to trained model .joblib file (must contain 'model' and 'offset_map')")
    req.add_argument("--validation-file", type=str, required=True,
                     help="Path to validation CSV file with features and target")
    req.add_argument("--results-dir", type=str, required=True,
                     help="Directory to save metrics, results CSV, and plot")

    opt = parser.add_argument_group('optional arguments')
    opt.add_argument("--target-col", type=str, default="age",
                     help="Name of the target column in validation data")
    opt.add_argument("--id-col", type=str, default="sampleID",
                     help="Name of the sample ID column")
    opt.add_argument("--gender-col", type=str, default="gender",
                     help="Name of the gender column")
    opt.add_argument("--no-plot", action='store_true',
                     help="Disable scatter plot generation")
    opt.add_argument("--seed", type=int, default=42,
                     help="Random seed for reproducibility")
    opt.add_argument("--verbose", action='store_true',
                     help="Enable verbose logging")
    opt.add_argument("--help-extended", action='store_true',
                     help="Show detailed help and exit")
    return parser.parse_args()


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=level,
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def main():
    args = parse_args()
    if args.help_extended:
        print(__doc__)
        sys.exit(0)

    setup_logging(args.verbose)
    logger = logging.getLogger()

    os.makedirs(args.results_dir, exist_ok=True)
    metrics_csv = os.path.join(args.results_dir, 'validation_metrics.csv')
    results_csv = os.path.join(args.results_dir, 'validation_results.csv')
    plot_path   = os.path.join(args.results_dir, 'validation_scatter.png')

    # Load trained model and offset map
    loaded = joblib.load(args.model_path)
    if not (isinstance(loaded, dict) and 'model' in loaded and 'offset_map' in loaded):
        logger.error("Model file must be a dict with 'model' and 'offset_map' keys")
        sys.exit(1)
    model = loaded['model']
    offset_map = loaded['offset_map']

    # Ensure offset_map is a pandas Series
    if isinstance(offset_map, pd.DataFrame):
        offset_series = offset_map.set_index('age')['offset']
    elif isinstance(offset_map, pd.Series):
        offset_series = offset_map
    else:
        logger.error("offset_map must be a pandas DataFrame or Series")
        sys.exit(1)

    # Load validation data
    df = pd.read_csv(args.validation_file)
    df['gender_encoded'] = df[args.gender_col].map({'Male':0,'Female':1}).fillna(-1)

    # Prepare features and target
    drop_cols = [args.id_col, args.target_col, args.gender_col]
    for col in ('age_bin','age_range'):
        if col in df.columns:
            drop_cols.append(col)
    X_val = df.drop(columns=drop_cols, errors='ignore')
    y_val = df[args.target_col]

    # Align validation features with training features
    if hasattr(model, 'booster_'):
        feat_names = model.booster_.feature_name()
    elif hasattr(model, 'feature_name_'):
        feat_names = model.feature_name_
    else:
        feat_names = X_val.columns.tolist()
    X_val = X_val.reindex(columns=feat_names, fill_value=0)

    # Generate raw predictions
    y_raw = model.predict(X_val)

    # Apply bias correction using offset_series
    ages = y_val.astype(int)
    offsets = offset_series.reindex(ages).fillna(0).to_numpy()
    y_pred = y_raw - offsets

    # Compute performance metrics
    y_true = y_val.to_numpy()
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)
    logger.info("Validation Metrics — MAE: %.3f, MSE: %.3f, R²: %.3f, Pearson R: %.3f", mae, mse, r2, r)

    # Save metrics
    pd.DataFrame([{'MAE':mae, 'MSE':mse, 'R2':r2, 'PearsonR':r}]).to_csv(metrics_csv, index=False)
    logger.info("Metrics saved to %s", metrics_csv)

    # Save detailed results
    results_df = pd.DataFrame({
        args.id_col:     df[args.id_col].values,
        'Actual_Age':    y_true,
        'Predicted_Age': y_pred,
        args.gender_col: df[args.gender_col].values
    })
    results_df.to_csv(results_csv, index=False)
    logger.info("Results saved to %s", results_csv)

    # Scatter plot by gender
    if not args.no_plot:
        plt.figure(figsize=(8,6))
        colors = {'Male':'blue','Female':'red'}
        for gender in df[args.gender_col].unique():
            mask = df[args.gender_col] == gender
            plt.scatter(
                y_pred[mask], y_true[mask], alpha=0.6,
                label=f"{gender}: MAE={mean_absolute_error(y_true[mask], y_pred[mask]):.2f}, "
                      f"R²={r2_score(y_true[mask], y_pred[mask]):.2f}",
                color=colors.get(gender)
            )
        mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        plt.plot([mn, mx], [mn, mx], '--', linewidth=1.5)
        plt.xlabel('Predicted Age')
        plt.ylabel('Actual Age')
        plt.title(f'Validation: Pred vs Actual (R²={r2:.3f})')
        plt.legend(title='Gender')
        plt.tight_layout()
        plt.savefig(plot_path)
        logger.info("Scatter plot saved to %s", plot_path)
        plt.show()

if __name__ == '__main__':
    main()
