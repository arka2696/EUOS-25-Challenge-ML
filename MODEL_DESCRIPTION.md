# EUOS25 Challenge — Model Description

Our submission to the EUOS25 Challenge employs a structure-based stacked ensemble framework to predict four binary optical property endpoints directly from molecular representations. The modeling pipeline is strictly limited to information derivable from canonical SMILES strings, ensuring that predictions are independent of any experimental metadata. Neither well identifiers, plate information, batch annotations, nor external databases are used at any stage of feature construction, model training, or inference. The training dataset consists of 68,972 molecules, each annotated with four binary labels, while the blind test set contains 29,420 molecules represented solely by their identifiers and canonical SMILES strings. The compound identifier is used exclusively for indexing and submission formatting and is never incorporated as a predictive feature.

## Feature Engineering

Feature engineering is performed using RDKit to transform each SMILES string into a high-dimensional molecular representation capturing structural and physicochemical information. Circular substructure information is encoded using Morgan fingerprints (Extended Connectivity Fingerprints) at two radii: ECFP4 (radius 2) and ECFP6 (radius 3), each represented as a 2,048-bit binary vector, yielding 4,096 dimensions in total. These fingerprints capture local atomic environments and neighbourhood patterns across multiple bond radii. Complementing these representations, 167-bit MACCS structural keys encode predefined substructure patterns, including aromatic systems and functional groups. In addition, a panel of 15 physicochemical descriptors is computed from the molecular graph:

| Descriptor | Description |
|---|---|
| MolWt | Molecular weight |
| MolLogP | Wildman–Crippen LogP (lipophilicity) |
| TPSA | Topological polar surface area |
| NumHDonors | Number of hydrogen bond donors |
| NumHAcceptors | Number of hydrogen bond acceptors |
| NumRotatableBonds | Number of rotatable bonds |
| NumAromaticRings | Count of aromatic rings |
| FractionCSP3 | Fraction of sp3-hybridised carbons |
| HeavyAtomCount | Total number of non-hydrogen atoms |
| RingCount | Total ring count |
| NHOHCount | Count of NH and OH groups |
| NOCount | Count of nitrogen and oxygen atoms |
| NumAliphaticRings | Number of aliphatic ring systems |
| NumSaturatedRings | Number of fully saturated rings |
| MolMR | Molar refractivity |

These descriptors are standardised using zero-mean, unit-variance scaling parameters computed exclusively on the training set and subsequently applied unchanged to the test set to prevent data leakage. The resulting feature vector per molecule comprises approximately 4,278 dimensions (4,096 Morgan bits + 167 MACCS bits + 15 descriptors).

## Model Architecture

### Base Learners (Level 1)

The predictive architecture is based on a two-level stacked ensemble. At the first level, three complementary model families are trained independently: Random Forest, Extra Trees, and LightGBM gradient boosting. Each model is implemented using scikit-learn or LightGBM and trained using 5-fold stratified cross-validation. Stratification is performed according to whether a sample contains any positive label across the four tasks, thereby preserving class distribution under severe imbalance conditions. For each fold and each task, a separate binary classifier is trained. The Random Forest model uses 400 trees with balanced subsampling, while the Extra Trees model uses 800 trees with analogous class weighting. The LightGBM model is trained with up to 5,000 boosting rounds, early stopping with a patience of 100 rounds, 127 leaves, feature_fraction and bagging_fraction set to 0.8, and scale_pos_weight calibrated to the negative-to-positive ratio for each task. Class weighting strategies are systematically applied across all base learners to address the pronounced class imbalance, particularly for the Fluorescence (>480 nm) endpoint with a positive rate of approximately 0.24%.

### Meta-Learner (Level 2 — Stacking)

At the second level, stacking is performed via a logistic regression meta-learner trained independently for each task. Out-of-fold (OOF) predictions from the three base models are used to construct a three-dimensional meta-feature vector per sample per endpoint. A logistic regression model (C = 1.0, class_weight = "balanced", solver = lbfgs) is trained using an additional 5-fold cross-validation procedure to generate unbiased meta-level OOF predictions. Finally, a logistic regression model is fitted on the complete OOF meta-feature set for inference on the test data. This approach enables the meta-learner to optimally weight the complementary inductive biases of tree-based bagging and gradient boosting methods rather than relying on simple averaging.

### Multi-Seed Ensembling

To further enhance robustness and reduce variance, the entire training and stacking pipeline is repeated across three independent random seeds (42, 52, and 62). Final test predictions are obtained by averaging the stacked probabilities across these seed-level ensembles.

## Blind Test Inference Pipeline

The blind test inference pipeline consists of SMILES parsing, RDKit-based feature extraction (Morgan fingerprints, MACCS keys, and standardised descriptors), generation of base model predictions averaged across cross-validation folds, meta-level logistic regression stacking, and final averaging across seeds to produce task-specific probability estimates.

## Confirmation: No Well/Plate Information Used

We confirm that:

1. No well or plate information is used as input features, for stratification, or at any other stage of the pipeline
2. The only input to the model is the SMILES string for each molecule
3. The compound ID column is used solely for indexing the submission file and is never used as a feature
4. Feature computation is entirely structure-based (fingerprints and physicochemical descriptors derived from the molecular graph)
5. There is no use of external databases, pre-computed assay results, or any information beyond what is derivable from the SMILES representation

## Software Environment

The software environment is implemented in Python 3.10, leveraging RDKit for molecular parsing and descriptor computation, scikit-learn for ensemble and meta-learning models, LightGBM for gradient boosting, and NumPy and Pandas for data handling.

## Cross-Validation Performance

Cross-validation performance, averaged across seeds:

| Task | ROC-AUC |
|------|---------|
| Fluorescence (>480 nm) | ~0.60 |
| Fluorescence (340/450 nm) | ~0.87 |
| Transmittance (450 nm) | ~0.65 |
| Transmittance (340 nm) | ~0.85 |

These results demonstrate strong predictive performance for the higher-signal tasks and moderate discrimination under extreme class imbalance.
