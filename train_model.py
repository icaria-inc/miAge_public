#!/usr/bin/env python3
"""
Train and evaluate an age prediction model using LightGBM and Optuna.

This script performs nested stratified cross-validation, hyperparameter tuning,
bias correction (always applied), and generates performance metrics and plots.

Usage:
    python train_model.py \
        --input-file path/to/data.csv \
        --results-dir results/ \
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
"""
import argparse
import os
import sys
import logging
import warnings

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import optuna
import numpy as np
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

def parse_args():
    from argparse import ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(
        description="Train age prediction model using LightGBM and Optuna",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        "--input-file", type=str, required=True,
        help="Path to input CSV file containing features and target"
    )
    required.add_argument(
        "--results-dir", type=str, required=True,
        help="Directory where outputs (models, plots, CSVs) will be saved"
    )

    optional = parser.add_argument_group('optional arguments')
    optional.add_argument("--target-col", type=str, default="age",
                          help="Name of the target column in the input data")
    optional.add_argument("--id-col", type=str, default="sampleID",
                          help="Name of the sample ID column")
    optional.add_argument("--gender-col", type=str, default="gender",
                          help="Name of the gender column (will be encoded)")
    optional.add_argument("--n-splits", type=int, default=5,
                          help="Number of folds for outer stratified CV")
    optional.add_argument("--inner-splits", type=int, default=5,
                          help="Number of folds for inner CV during tuning")
    optional.add_argument("--quantile-bins", type=int, default=5,
                          help="Number of quantile bins for age stratification")
    optional.add_argument("--n-trials", type=int, default=100,
                          help="Number of Optuna trials for hyperparameter tuning")
    optional.add_argument("--early-stopping", type=int, default=20,
                          help="Early stopping rounds for LightGBM")
    optional.add_argument("--metric", type=str, default="rmse", choices=["rmse","mae"],
                          help="Evaluation metric for LightGBM")
    optional.add_argument("--no-plot", action='store_true',
                          help="Do not generate diagnostic scatter plot")
    optional.add_argument("--seed", type=int, default=42,
                          help="Random seed for reproducibility")
    optional.add_argument("--verbose", action='store_true',
                          help="Enable verbose logging")
    optional.add_argument("--help-extended", action='store_true',
                          help="Show detailed help (full docstring) and exit")
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
    model_output = os.path.join(args.results_dir, 'best_model.joblib')
    bias_plot_path = os.path.join(args.results_dir, 'bias_plot.png')
    combined_csv = os.path.join(args.results_dir, 'combined_results.csv')
    fi_csv = os.path.join(args.results_dir, 'feature_importances.csv')

    # Load and prepare data
    data = pd.read_csv(args.input_file)
    data['gender_encoded'] = data[args.gender_col].map({'Male':0,'Female':1}).fillna(-1)
    data['age_bin'] = pd.qcut(
        data[args.target_col], args.quantile_bins,
        labels=False, duplicates='drop'
    )

    drop_cols = [args.id_col, args.target_col, args.gender_col, 'age_bin']
    if 'age_range' in data.columns:
        drop_cols.append('age_range')
    X = data.drop(columns=drop_cols)
    y = data[args.target_col]
    bins = data['age_bin'].values

    outer_cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    fold_metrics = []
    all_results = []
    feature_importances = []
    best_score = -np.inf
    best_offset_map = None

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, bins), start=1):
        logger.info(f"Starting fold {fold}/{args.n_splits}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        train_bins = bins[train_idx]

        def objective(trial):
            params = {
                'objective':'regression','metric':args.metric,
                'learning_rate':trial.suggest_loguniform('learning_rate',1e-3,1e-1),
                'max_depth':trial.suggest_int('max_depth',3,15),
                'num_leaves':trial.suggest_int('num_leaves',20,128),
                'subsample':trial.suggest_uniform('subsample',0.5,1.0),
                'colsample_bytree':trial.suggest_uniform('colsample_bytree',0.5,1.0),
                'reg_alpha':trial.suggest_loguniform('reg_alpha',1e-3,10.0),
                'reg_lambda':trial.suggest_loguniform('reg_lambda',1e-3,10.0),
                'random_state':args.seed,'verbosity':-1
            }
            inner_cv = StratifiedKFold(n_splits=args.inner_splits, shuffle=True, random_state=args.seed)
            scores, iters = [], []
            for tr_i, val_i in inner_cv.split(X_train, train_bins):
                mdl = lgb.LGBMRegressor(**params)
                mdl.fit(
                    X_train.iloc[tr_i], y_train.iloc[tr_i],
                    eval_set=[(X_train.iloc[val_i], y_train.iloc[val_i])],
                    eval_metric=args.metric,
                    callbacks=[lgb.callback.early_stopping(args.early_stopping)]
                )
                preds_val = mdl.predict(X_train.iloc[val_i])
                scores.append(r2_score(y_train.iloc[val_i], preds_val))
                iters.append(mdl.best_iteration_ or mdl.n_estimators)
            trial.set_user_attr('best_iter', int(np.percentile(iters,25)))
            return np.mean(scores)

        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=args.seed), pruner=MedianPruner(n_warmup_steps=5))
        study.optimize(objective, n_trials=args.n_trials, n_jobs=-1)

        params = study.best_params.copy()
        params.update({
            'n_estimators': study.best_trial.user_attrs['best_iter'],
            'objective':'regression','metric':args.metric,
            'random_state':args.seed,'verbosity':-1
        })
        logger.debug(f"Fold {fold} tuned params: {params}")

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)

        # Bias-correction: always applied
        y_tr_pred = model.predict(X_train)
        df_tr = pd.DataFrame({'age': y_train, 'pred': y_tr_pred})
        df_tr['residual'] = df_tr['pred'] - df_tr['age']
        age_offset = df_tr.groupby('age')['residual'].mean().reset_index(name='offset')
        all_ages = pd.DataFrame({'age': np.arange(data[args.target_col].min(), data[args.target_col].max()+1)})
        offset_map = (
            all_ages
            .merge(age_offset, on='age', how='left')
            .assign(offset=lambda d: d.offset.interpolate().fillna(0))
            .set_index('age')['offset']
        )
        preds_raw = model.predict(X_test)
        preds     = preds_raw - offset_map.loc[y_test].values

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        r2  = r2_score(y_test, preds)
        logger.info(f"Fold {fold} — MAE: {mae:.3f}, MSE: {mse:.3f}, R²: {r2:.3f}")
        fold_metrics.append({'fold':fold,'MAE':mae,'MSE':mse,'R2':r2})

        all_results.append(pd.DataFrame({
            args.id_col: data.iloc[test_idx][args.id_col].values,
            'Actual': y_test.values,
            'Predicted': preds,
            args.gender_col: data.iloc[test_idx][args.gender_col].values
        }))
        feature_importances.append(pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_,
            'fold': fold
        }))

        if r2 > best_score:
            best_score = r2
            best_offset_map = offset_map
            joblib.dump({'model': model, 'offset_map': best_offset_map}, model_output)
            logger.info(f"Saved new best model (R²={r2:.3f}) with offset map to {model_output}")

    pd.DataFrame(fold_metrics).to_csv(os.path.join(args.results_dir,'cv_metrics.csv'), index=False)
    pd.concat(feature_importances).groupby('feature')['importance'].mean().reset_index() \
        .sort_values('importance',ascending=False).to_csv(fi_csv,index=False)
    pd.concat(all_results).to_csv(combined_csv,index=False)

    if not args.no_plot:
        overall = pd.concat(all_results)
        plt.figure(figsize=(8,6))
        for g in overall[args.gender_col].unique():
            sub = overall[overall[args.gender_col]==g]
            plt.scatter(sub['Predicted'], sub['Actual'], alpha=0.6, label=g)
        mn, mx = overall[['Actual','Predicted']].values.min(), overall[['Actual','Predicted']].values.max()
        plt.plot([mn, mx], [mn, mx], '--')
        plt.xlabel('Predicted Age')
        plt.ylabel('Actual Age')
        plt.title(f"Pred vs Actual (Overall R²={r2_score(overall['Actual'],overall['Predicted']):.3f})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(bias_plot_path)
        logger.info(f"Plot saved to {bias_plot_path}")

if __name__ == '__main__':
    main()
