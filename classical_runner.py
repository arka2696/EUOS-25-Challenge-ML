#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

from euos25.features.classical import compute_feature_matrix
from euos25.ensemble import average_model_predictions
from euos25.eval import evaluate_predictions

from euos25.models.classical import (
    train_random_forest_cv,
    train_extra_trees_cv,
    train_pls_regression_cv,
)

# Optional boosting
try:
    from euos25.models.classical import train_lightgbm_cv
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from euos25.models.classical import train_catboost_cv
    HAS_CB = True
except Exception:
    HAS_CB = False

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression


# -------------------
# Config
# -------------------
TRAIN_CSV = "merged_train.csv"
TEST_CSV  = "cleaned_test.csv"

SMILES_COL = "SMILES"
ID_COL = "ID"
LABEL_COLS = ["Fluorescence480", "Fluorescence340_450", "Transmittance450", "Transmittance340"]

OUT_EVAL = "outputs/eval"
OUT_SUB = "outputs/submissions"
os.makedirs("outputs", exist_ok=True)
os.makedirs(OUT_EVAL, exist_ok=True)
os.makedirs(OUT_SUB, exist_ok=True)

# Top-5 knobs (safe defaults)
SEEDS = [42, 52, 62]  # set to [42] for quick runs
N_SPLITS = 5

# Feature settings
FEATURE_KW = dict(
    include_maccs=True,
    include_descriptors=True,
    radii=(2, 3),          # ECFP4 + ECFP6
    n_bits=2048,
    use_counts=False,      # try True later (often helps LGBM)
    descriptor_variance_cutoff=0.0,
    standardize_descriptors=True,
    use_chirality=False,   # can try True as an experiment
)

# Model family toggles
USE_RF = True
USE_ET = True
USE_PLS = False           # usually weak vs boosting; keep False unless it helps you
USE_LGBM = HAS_LGBM
USE_CB = False            # enable only if installed

# Meta strategy
USE_STACKING = True       # main Top-5 method
META_C = 1.0              # LogisticRegression regularization strength


def _stratify_label(y: np.ndarray) -> np.ndarray:
    return (y.sum(axis=1) > 0).astype(int)


def _stack_meta_oof_and_models(
    base_oof: np.ndarray,
    y: np.ndarray,
    strat_label: np.ndarray,
    n_splits: int,
    seed: int,
):
    """
    Proper stacking with 2nd-level OOF:
    - base_oof: (n_samples, n_models, n_tasks)
    - returns meta_oof: (n_samples, n_tasks) + final_meta_models (per task)
    """
    n_samples, n_models, n_tasks = base_oof.shape
    meta_oof = np.full((n_samples, n_tasks), np.nan, dtype=np.float32)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # For each task, build honest meta-OOF via CV on base OOF features
    for t in range(n_tasks):
        X_meta = base_oof[:, :, t]  # (n_samples, n_models)
        y_t = y[:, t].astype(int)

        for fold, (tr, va) in enumerate(skf.split(X_meta, strat_label)):
            clf = LogisticRegression(
                C=META_C,
                class_weight="balanced",
                max_iter=2000,
                solver="lbfgs",
            )
            clf.fit(X_meta[tr], y_t[tr])
            meta_oof[va, t] = clf.predict_proba(X_meta[va])[:, 1].astype(np.float32)

    # Fit final meta models on full base OOF (for test inference)
    final_meta = []
    for t in range(n_tasks):
        X_meta = base_oof[:, :, t]
        y_t = y[:, t].astype(int)
        clf = LogisticRegression(
            C=META_C,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
        )
        clf.fit(X_meta, y_t)
        final_meta.append(clf)

    if np.isnan(meta_oof).any():
        raise RuntimeError("Meta OOF contains NaNs — stacking CV did not fill everything.")

    return meta_oof, final_meta


def main():
    # -------------------
    # 1) Load data
    # -------------------
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    missing = [c for c in [ID_COL, SMILES_COL, *LABEL_COLS] if c not in train_df.columns]
    if missing:
        raise ValueError(f"Training CSV missing columns: {missing}")

    y = train_df[LABEL_COLS].astype(int).values
    strat_label = _stratify_label(y)

    print("Train:", train_df.shape, " Test:", test_df.shape)
    print("Positives:", dict(zip(LABEL_COLS, y.sum(axis=0).tolist())))
    print("HAS_LGBM:", HAS_LGBM, "HAS_CB:", HAS_CB)

    # -------------------
    # 2) Compute features (TRAIN fit transform -> apply to TEST)
    # -------------------
    print("\nComputing features for TRAIN (fit descriptor transform)...")
    X_train, feat_names, desc_tf = compute_feature_matrix(
        train_df,
        smiles_col=SMILES_COL,
        return_descriptor_transform=True,
        **FEATURE_KW,
    )

    print("Computing features for TEST (apply TRAIN transform)...")
    X_test, _ = compute_feature_matrix(
        test_df,
        smiles_col=SMILES_COL,
        descriptor_transform=desc_tf,
        **FEATURE_KW,
    )

    np.save("outputs/X_train.npy", X_train)
    np.save("outputs/X_test.npy", X_test)

    # -------------------
    # 3) Multi-seed training + ensembling
    # -------------------
    test_pred_accum = np.zeros((X_test.shape[0], len(LABEL_COLS)), dtype=np.float32)

    for seed in SEEDS:
        print(f"\n======================")
        print(f"Seed = {seed}")
        print(f"======================")

        # --- Train base families (OOF + models)
        families = []  # list of (name, oof, models)

        if USE_RF:
            print("Training RF...")
            rf_models, rf_aucs, rf_oof = train_random_forest_cv(
                X_train, y, LABEL_COLS, n_estimators=400, n_splits=N_SPLITS, random_state=seed
            )
            families.append(("rf", rf_oof, rf_models))

        if USE_ET:
            print("Training ET...")
            et_models, et_aucs, et_oof = train_extra_trees_cv(
                X_train, y, LABEL_COLS, n_estimators=800, n_splits=N_SPLITS, random_state=seed
            )
            families.append(("et", et_oof, et_models))

        if USE_PLS:
            print("Training PLS...")
            pls_models, pls_aucs, pls_oof = train_pls_regression_cv(
                X_train, y, LABEL_COLS, n_components=10, n_splits=N_SPLITS, random_state=seed
            )
            families.append(("pls", pls_oof, pls_models))

        if USE_LGBM:
            print("Training LGBM...")
            lgb_models, lgb_aucs, lgb_oof = train_lightgbm_cv(
                X_train,
                y,
                LABEL_COLS,
                n_splits=N_SPLITS,
                random_state=seed,
                # mild tuning defaults that often help:
                num_leaves=127,
                min_data_in_leaf=50,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=1,
            )
            families.append(("lgb", lgb_oof, lgb_models))

        if USE_CB and HAS_CB:
            print("Training CatBoost...")
            cb_models, cb_aucs, cb_oof = train_catboost_cv(
                X_train, y, LABEL_COLS, n_splits=N_SPLITS, random_state=seed
            )
            families.append(("cb", cb_oof, cb_models))

        if len(families) == 0:
            raise RuntimeError("No model families enabled.")

        # --- Build base OOF cube: (n_samples, n_models, n_tasks)
        base_names = [n for (n, _, _) in families]
        base_oof = np.stack([oof for (_, oof, _) in families], axis=1)  # models axis

        if np.isnan(base_oof).any():
            raise RuntimeError("Base OOF contains NaNs — a fold likely failed.")

        # -------------------
        # 4) Meta strategy: stacking (recommended)
        # -------------------
        if USE_STACKING and (len(base_names) >= 2):
            print("Stacking meta-model on OOF predictions...")
            meta_oof, meta_models = _stack_meta_oof_and_models(
                base_oof=base_oof,
                y=y,
                strat_label=strat_label,
                n_splits=N_SPLITS,
                seed=seed + 123,
            )

            # Evaluate honest meta OOF
            metrics = evaluate_predictions(y, meta_oof, LABEL_COLS, save_dir=OUT_EVAL, prefix=f"seed{seed}_meta_")
            pd.DataFrame(metrics).T.to_csv(os.path.join(OUT_EVAL, f"seed{seed}_meta_metrics.csv"))
            print(pd.DataFrame(metrics).T)

            # Predict test: compute base test preds, then meta-predict
            base_test_list = []
            for name, _, models in families:
                pred = average_model_predictions(models, X_test, LABEL_COLS)  # (n_test, n_tasks)
                base_test_list.append(pred)
            base_test = np.stack(base_test_list, axis=1)  # (n_test, n_models, n_tasks)

            test_pred_seed = np.zeros((X_test.shape[0], len(LABEL_COLS)), dtype=np.float32)
            for t in range(len(LABEL_COLS)):
                X_meta_test = base_test[:, :, t]
                test_pred_seed[:, t] = meta_models[t].predict_proba(X_meta_test)[:, 1].astype(np.float32)

        else:
            # fallback: simple average
            print("Falling back to simple mean ensemble (no stacking).")
            test_pred_seed = np.mean(
                np.stack([average_model_predictions(models, X_test, LABEL_COLS) for (_, _, models) in families], axis=0),
                axis=0
            ).astype(np.float32)

        test_pred_accum += test_pred_seed

    # Average over seeds
    test_pred = (test_pred_accum / float(len(SEEDS))).astype(np.float32)

    # -------------------
    # 5) Write submission (internal)
    # -------------------
    sub = pd.DataFrame({ID_COL: test_df[ID_COL].values})
    for i, col in enumerate(LABEL_COLS):
        sub[col] = test_pred[:, i]

    out_path = os.path.join(OUT_SUB, "submission_classical.csv")
    sub.to_csv(out_path, index=False)
    print("\nSaved:", out_path)

    # -------------------
    # 6) Write submission (EUOS upload format)
    # Required order:
    # Transmittance(340),Transmittance(450),Fluorescence(340/450),Fluorescence(>480)
    # -------------------
    df_fixed = sub[["Transmittance340", "Transmittance450", "Fluorescence340_450", "Fluorescence480"]].copy()
    df_fixed.columns = ["Transmittance(340)", "Transmittance(450)", "Fluorescence(340/450)", "Fluorescence(>480)"]
    out_path2 = os.path.join(OUT_SUB, "submission_classical_EUOS25.csv")
    df_fixed.to_csv(out_path2, index=False)
    print("Saved (EUOS format):", out_path2)


if __name__ == "__main__":
    main()
