# EUOS-25-Challenge-ML

Machine learning pipeline for molecular optical property prediction, developed for the **EU-OPENSCREEN / SLAS EUOS25 Challenge**.

Predicts **absorption** and **fluorescence** properties of ~69,000 small molecules using only their **SMILES strings** — no well/plate metadata, no external databases, no experimental context.

---

## Challenge Overview

The competition requires predicting four binary optical property endpoints:

| Task | Endpoint | Positive Rate |
|------|----------|---------------|
| **Transmittance (340 nm)** | Absorption at 340 nm | 5.6% |
| **Transmittance (450 nm)** | Absorption at 450–679 nm avg | 1.5% |
| **Fluorescence (340/450 nm)** | Excitation 340 nm → Emission 450 nm | 16.7% |
| **Fluorescence (>480 nm)** | Excitation 480 nm → Emission 540/598/610 nm | **0.24%** |

All predictions are evaluated using **ROC-AUC**, averaged across subtasks.

---

## Pipeline Architecture

```
5 Raw CSVs (data/)
    │
    ▼
merge_csv.py ──► merged_train.csv + cleaned_test.csv
    │
    ▼
classical_runner.py
    │
    ├── Feature Engineering
    │     Morgan FP (ECFP4 + ECFP6) + MACCS Keys + 15 RDKit Descriptors
    │     → 4,278-dimensional feature vector per molecule
    │
    ├── Multi-Seed Training (Seeds: 42, 52, 62)
    │     5-Fold Stratified CV per seed
    │     Base Models: RandomForest + ExtraTrees + LightGBM
    │
    ├── Stacking Meta-Model
    │     LogisticRegression per task, trained on OOF predictions
    │
    ├── Evaluation (ROC-AUC, PR-AUC, Confusion Matrices)
    │
    └── Seed-Averaged Submission CSVs
```

---

## Feature Engineering

Each SMILES string is converted into a **4,278-dimensional feature vector**:

| Feature | Dimensions | Description |
|---------|------------|-------------|
| Morgan FP (radius 2) | 2,048 bits | ECFP4 — local atomic environments |
| Morgan FP (radius 3) | 2,048 bits | ECFP6 — extended neighbourhood |
| MACCS Keys | 167 bits | Predefined substructure patterns |
| RDKit Descriptors | 15 | MolWt, LogP, TPSA, aromaticity, ring counts, etc. |

Descriptors are standardised using train-set statistics only (no data leakage).

Source: `euos25/features/classical.py`

---

## Model Architecture

### Level 1 — Base Learners

| Model | Trees | Key Settings |
|-------|-------|-------------|
| **RandomForest** | 400 | `class_weight="balanced_subsample"` |
| **ExtraTrees** | 800 | `class_weight="balanced_subsample"` |
| **LightGBM** | 5,000 max (early stop @ 100) | `num_leaves=127`, `scale_pos_weight=neg/pos` |

Each model trains independently via 5-fold stratified CV, producing out-of-fold (OOF) predictions.

### Level 2 — Stacking Meta-Learner

A **LogisticRegression** (per task) is trained on the 3 base model OOF probabilities as meta-features, using balanced class weights and a second 5-fold CV for honest meta-level estimates.

### Multi-Seed Ensembling

The entire pipeline repeats across **3 seeds** (42, 52, 62). Final test probabilities are the seed-averaged stacked predictions.

---

## Cross-Validation Results

| Task | Seed 42 | Seed 52 | Seed 62 | Avg |
|------|---------|---------|---------|-----|
| Fluorescence (>480 nm) | 0.580 | 0.603 | 0.606 | ~0.60 |
| Fluorescence (340/450 nm) | **0.870** | **0.871** | **0.869** | **~0.87** |
| Transmittance (450 nm) | 0.648 | 0.645 | 0.655 | ~0.65 |
| Transmittance (340 nm) | **0.844** | **0.845** | **0.846** | **~0.85** |

> Fluorescence (>480 nm) has the lowest AUC due to extreme class imbalance (164 positives in 69K samples).

---

## Project Structure

```
EUOS25/
├── classical_runner.py            # Main pipeline entry point
├── classical_runner.sbatch        # SLURM job submission script
├── merge_csv.py                   # Merges raw CSVs into unified dataset
├── submission_ready.sh            # Submission preparation helper
│
├── euos25/                        # Modular Python package
│   ├── __init__.py
│   ├── config.py                  # Euos25Config dataclass
│   ├── io.py                      # Data loading & CSV merge utilities
│   ├── ensemble.py                # Fold-averaged inference
│   ├── eval.py                    # Metrics, ROC/PR curves, confusion matrices
│   ├── features/
│   │   ├── classical.py           # Morgan FP + MACCS + RDKit descriptors
│   │   ├── augmentation.py        # SMILES augmentation (tautomers, randomized)
│   │   └── graphs.py              # Graph-based featurization (for GNN)
│   └── models/
│       ├── classical.py           # RF, ET, LightGBM, XGBoost, CatBoost
│       └── gnn.py                 # GNN model definitions
│
├── euos25_full_suite.py           # Original monolithic pipeline (all-in-one)
├── euos25_comprehensive_models.py # Extended model experiments
├── euos25_comprehensive_eda.py    # Standalone EDA script
├── inspect_merged_euos25.py       # Data inspection utilities
│
├── data/                          # Raw challenge CSVs (5 files)
├── outputs/
│   ├── eval/                      # Per-seed metrics, ROC/PR/CM plots
│   ├── eda/                       # EDA plots and charts
│   └── submissions/               # Final submission CSVs
│
├── singularity/
│   ├── euos25_classical.def       # Singularity container definition
│   └── euos25_GPU.def             # GPU container definition
│
├── merge_report/                  # Merge diagnostics (histograms, correlations)
├── env.yml                        # Conda environment specification
├── requirements.txt               # pip requirements
├── MODEL_DESCRIPTION.md           # Formal model description document
└── PIPELINE_WALKTHROUGH.md        # Detailed pipeline walkthrough
```

---

## How to Run

### 1. Set Up Environment

**Using Conda:**

```bash
conda env create -f env.yml
conda activate euos25_env
```

**Using pip:**

```bash
pip install -r requirements.txt
```

### 2. Merge Datasets

Combine the 5 raw training/test CSVs into unified files:

```bash
python merge_csv.py
```

This produces `merged_train.csv` (68,972 rows × 6 columns) and `cleaned_test.csv` (29,420 rows).

### 3. Run the Pipeline

**Locally:**

```bash
python classical_runner.py
```

**On HPC (SLURM + Singularity):**

```bash
sbatch classical_runner.sbatch
```

The SBATCH script allocates 80 CPUs and 128 GB RAM, launches the Singularity container (`singularity/euos25_classical.sif`), and runs the pipeline inside it. Logs are written to `logs/`.

**Quick single-seed run** — edit `classical_runner.py` line 50:

```python
SEEDS = [42]  # instead of [42, 52, 62]
```

### 4. Outputs

After a full run, you will find:

| Directory | Contents |
|-----------|----------|
| `outputs/eval/` | Per-seed ROC curves, PR curves, confusion matrices, metrics CSVs |
| `outputs/submissions/` | `submission_classical_EUOS25.csv` (challenge format) |
| `outputs/eda/` | EDA plots (class distributions, correlations, functional groups) |

---

## Key Tuning Parameters

| Parameter | File | Default | Notes |
|-----------|------|---------|-------|
| `SEEDS` | `classical_runner.py` | `[42, 52, 62]` | More seeds = more robust, slower |
| `N_SPLITS` | `classical_runner.py` | `5` | Cross-validation folds |
| `radii` | `classical_runner.py` | `(2, 3)` | ECFP4 + ECFP6; try `(2, 3, 4)` for ECFP8 |
| `n_bits` | `classical_runner.py` | `2048` | Fingerprint length; try 4096 |
| `use_counts` | `classical_runner.py` | `False` | Count-based FP (often helps LightGBM) |
| `USE_CB` | `classical_runner.py` | `False` | Enable CatBoost base learner |
| `META_C` | `classical_runner.py` | `1.0` | Stacking LR regularization |

---

## Confirmation: No Experimental Metadata Used

- The **only input** to the model is the SMILES string for each molecule
- No well/plate information, batch annotations, or external databases are used
- The compound ID column is used solely for submission indexing
- All features are derived from the molecular graph (fingerprints + physicochemical descriptors)

---

## References

- [EU-OPENSCREEN](https://www.eu-openscreen.eu/) Assay Documentation
- SLAS EUOS/SLAS Joint Challenge
- DeepChem / Chemprop / OGB MoleculeNet
- Joung et al., *Deep Learning Optical Spectroscopy*, JACS Au (2021)
- Nguyen et al., *Hybrid GNN Architecture for Spectral Prediction*, ACS Omega (2025)
