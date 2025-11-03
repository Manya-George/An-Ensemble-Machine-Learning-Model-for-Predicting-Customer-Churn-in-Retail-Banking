# prepare_model.py
"""
Enhanced prepare_model.py
-------------------------
- Loads dataset (columns auto-detected from CSV)
- Performs univariate statistical tests and optional visual exploration
- Builds preprocessing pipeline (imputation, scaling, one-hot)
- Applies SMOTE on preprocessed training data
- Performs L1-based feature selection
- Trains CatBoostClassifier
- Computes SHAP (global + keeps ability to compute local explanations)
- Produces combined ranking of features (statistical + tree-based)
- Saves all artifacts (model, preprocessor, selector, SHAP explainer,
  feature maps, statistical tables, combined ranking) to joblib file.
"""

import warnings
warnings.filterwarnings('ignore')

import os
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier

import shap

# plotting libs (only used if plot=True)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configurable defaults
DEFAULT_CONFIG = {
    "target_col": "Exited",      # preferred target column name; if missing, last column used
    "random_state": 42,
    "test_size": 0.2,
    "model_output_path": "catboost_churn_model.joblib",
    "plot": False,               # set True to show plots during analysis
    "catboost_params": {
        "iterations": 1000,
        "depth": 8,
        "learning_rate": 0.03,
        "l2_leaf_reg": 3,
        "random_seed": 42,
        "verbose": 100
    }
}


# ---------------------------
# Utility & Data Load
# ---------------------------
def load_data(filepath: str, target_col: Optional[str] = None) -> (pd.DataFrame, str):
    """
    Load CSV and determine target column automatically if needed.
    Returns (df, target_col)
    """
    print(f"Loading data from '{filepath}'...")
    df = pd.read_csv(filepath)
    if df.shape[0] == 0:
        raise ValueError("Dataframe is empty")

    if target_col and target_col in df.columns:
        tgt = target_col
    elif "Exited" in df.columns:
        tgt = "Exited"
    else:
        # default: use last column as target with warning
        tgt = df.columns[-1]
        print(f"Warning: target_col not found; using last column '{tgt}' as target.")

    print(f"Dataset shape: {df.shape}. Using target column: '{tgt}'")
    return df, tgt


# ---------------------------
# 1) UNIVARIATE / STATISTICAL ANALYSIS
# ---------------------------
def univariate_analysis(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Perform univariate stats (t-test for numeric, chi-square for categorical)
    Returns a DataFrame summarizing p-values and effects.
    """
    print("\nRunning univariate statistical analysis...")
    features = [c for c in df.columns if c != target_col]
    results = []

    for feat in features:
        ser = df[feat]
        if pd.api.types.is_numeric_dtype(ser):
            churned = df[df[target_col] == 1][feat].dropna()
            not_churned = df[df[target_col] == 0][feat].dropna()
            # if one of groups is empty, skip
            if len(churned) < 2 or len(not_churned) < 2:
                continue
            t_stat, p_value = stats.ttest_ind(churned, not_churned, equal_var=False, nan_policy='omit')
            results.append({
                "Feature": feat,
                "Type": "Numerical",
                "Churned_Mean": churned.mean(),
                "NotChurned_Mean": not_churned.mean(),
                "Difference": churned.mean() - not_churned.mean(),
                "T_Statistic": float(t_stat),
                "P_Value": float(p_value),
                "Significant": p_value < 0.05
            })
        else:
            # categorical
            contingency = pd.crosstab(df[feat].fillna("MISSING"), df[target_col])
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                # not enough variation
                continue
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            results.append({
                "Feature": feat,
                "Type": "Categorical",
                "Chi2_Statistic": float(chi2),
                "P_Value": float(p_value),
                "Significant": p_value < 0.05
            })

    res_df = pd.DataFrame(results).sort_values('P_Value', ascending=True).reset_index(drop=True)
    print(f"Univariate analysis computed for {len(res_df)} features.")
    return res_df


# ---------------------------
# 2) VISUAL EXPLORATION (optional)
# ---------------------------
def visual_exploration(df: pd.DataFrame, target_col: str, figsize=(14, 8)):
    """
    Optional: show distribution plots and correlation heatmap.
    """
    sns.set_palette("husl")
    plt.style.use('seaborn-v0_8-darkgrid')

    print("\nGenerating visual exploration plots...")
    features = [c for c in df.columns if c != target_col]
    numerical = df[features].select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in features if c not in numerical]

    # Numerical distributions by target
    if numerical:
        n_cols = 3
        n_rows = (len(numerical) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten()
        for i, feat in enumerate(numerical):
            ax = axes[i]
            churned = df[df[target_col] == 1][feat].dropna()
            not_churned = df[df[target_col] == 0][feat].dropna()
            ax.hist(not_churned, bins=30, alpha=0.6, density=True, label='Not churned')
            ax.hist(churned, bins=30, alpha=0.6, density=True, label='Churned')
            ax.set_title(feat)
            ax.legend()
        for j in range(len(numerical), len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.show()

    # Categorical churn rates
    if categorical:
        for feat in categorical:
            plt.figure(figsize=(8, 4))
            rates = df.groupby(feat)[target_col].agg(['mean', 'count']).sort_values('mean', ascending=False)
            rates['mean'].plot(kind='bar')
            plt.title(f'Churn Rate by {feat}')
            plt.ylabel('Churn Rate')
            plt.show()

    # Correlation heatmap for numerical features
    if numerical:
        corr = df[numerical + [target_col]].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0)
        plt.title('Correlation matrix (incl. target)')
        plt.show()


# ---------------------------
# 3) PREPROCESSING & PIPELINE
# ---------------------------
def build_preprocessor(X: pd.DataFrame):
    """
    Build and return a ColumnTransformer preprocessor that:
    - imputes numeric -> median, scales
    - imputes categorical -> most frequent, one-hot encodes
    Also returns the list of numeric and categorical columns used.
    """
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # OneHotEncoder config for sklearn>=1.2
    import sklearn
    from packaging import version
    if version.parse(sklearn.__version__) >= version.parse("1.2"):
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe)
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ], remainder='drop')

    return preprocessor, num_cols, cat_cols


# ---------------------------
# 4) TRAINING + SMOTE + SELECTION + CATBOOST
# ---------------------------
def train_model(X: pd.DataFrame, y: pd.Series, config: Dict[str, Any]):
    """
    Full training routine:
    - build preprocessor and fit_transform training data
    - apply SMOTE (on processed numeric + OHE features)
    - L1-based feature selection on balanced data
    - train CatBoost on selected features
    Returns artifacts dictionary
    """
    RANDOM_STATE = config.get("random_state", 42)
    cb_params = config.get("catboost_params", DEFAULT_CONFIG['catboost_params'])

    # Preprocessor
    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    print("Fitting preprocessor...")
    X_proc = preprocessor.fit_transform(X)

    # Construct processed feature names
    ohe = preprocessor.named_transformers_['cat'].named_steps['ohe']
    ohe_feature_names = []
    if hasattr(ohe, "get_feature_names_out"):
        ohe_feature_names = ohe.get_feature_names_out(cat_cols)
    else:
        # sklearn <1.0 fallback
        ohe_feature_names = ohe.get_feature_names(cat_cols)

    processed_feature_names = np.concatenate([num_cols, ohe_feature_names])
    processed_feature_names = [str(f) for f in processed_feature_names]

    print(f"Processed feature count: {len(processed_feature_names)}")

    # SMOTE on processed X
    print("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_bal, y_bal = smote.fit_resample(X_proc, y)
    print(f"Class distribution after SMOTE: {np.bincount(y_bal)}")

    # Feature selection with L1-penalized logistic regression
    print("Performing L1-based feature selection (LogisticRegression)...")
    sel_est = LogisticRegression(penalty="l1", solver="saga", max_iter=5000, random_state=RANDOM_STATE)
    sel_est.fit(X_bal, y_bal)
    selector = SelectFromModel(sel_est, prefit=True, threshold="median")
    X_bal_sel = selector.transform(X_bal)
    selected_mask = selector.get_support()
    selected_features = [f for f, keep in zip(processed_feature_names, selected_mask) if keep]
    print(f"Selected {len(selected_features)} features after selection")

    # Train CatBoost on selected features
    print("Training CatBoostClassifier...")
    model = CatBoostClassifier(**cb_params)
    model.fit(X_bal_sel, y_bal)

    # Build a small helper to transform arbitrary raw X to selected feature array:
    def process_and_select(X_raw: pd.DataFrame):
        Xp = preprocessor.transform(X_raw)
        return selector.transform(Xp)

    artifacts = {
        "model": model,
        "preprocessor": preprocessor,
        "selector": selector,
        "process_and_select_func": process_and_select,
        "processed_feature_names": processed_feature_names,
        "selected_features": selected_features,
        "original_num_cols": num_cols,
        "original_cat_cols": cat_cols,
        "smote": smote
    }
    return artifacts


# ---------------------------
# 5) SHAP INTERPRETABILITY
# ---------------------------
def compute_shap_and_importances(model: CatBoostClassifier, X_selected: np.ndarray, selected_features: list):
    """
    Compute SHAP values and global mean(|shap|) importances for the selected features.
    Returns a DataFrame of {feature, mean_abs_shap} sorted descending and the raw shap_values.
    """
    print("Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_selected)
    # shap_values shape: (n_samples, n_features)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": selected_features,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return explainer, shap_values, shap_df


# ---------------------------
# 6) COMBINED RANKING (statistical + catboost/shap)
# ---------------------------
def build_combined_ranking(stat_df: pd.DataFrame, shap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine univariate statistical ranking (p-values) and SHAP importance to produce a combined score.
    Returns a DataFrame sorted by combined score (descending).
    """
    print("Building combined ranking of features...")
    # stat_df may have entries for numeric & categorical; stat_df.Feature names match original column names.
    # shap_df.feature are processed selected feature names (which may be OHE features)
    # Strategy:
    #  - Map processed features to original features by prefix matching: e.g., "Geography_Germany" -> "Geography"
    #  - Aggregate shap mean_abs_shap per original feature (sum)
    #  - Combine with -log(p-value) or inverse rank from stats (if available)
    stat_map = {}
    if stat_df is not None and not stat_df.empty:
        # lower p-value -> higher score
        stat_df_local = stat_df.copy()
        stat_df_local["P_Value"] = stat_df_local["P_Value"].fillna(1.0)
        stat_df_local["stat_score"] = -np.log10(stat_df_local["P_Value"].clip(lower=1e-12))
        for _, r in stat_df_local.iterrows():
            stat_map[r["Feature"]] = float(r["stat_score"])

    # Map processed features back to original (by splitting OHE name on '_' and taking prefix)
    shap_df_local = shap_df.copy()
    # Try mapping: if feature contains one of original col names exactly at start -> map; else try split by '_'
    def infer_origin(proc_name):
        if "_" in proc_name:
            return proc_name.split("_")[0]
        else:
            return proc_name

    shap_df_local["orig_feature"] = shap_df_local["feature"].apply(infer_origin)
    shap_agg = shap_df_local.groupby("orig_feature")["mean_abs_shap"].sum().reset_index()

    # Build combined scores
    combined_scores = []
    all_features = set(list(stat_map.keys()) + shap_agg["orig_feature"].tolist())

    shap_dict = dict(zip(shap_agg["orig_feature"], shap_agg["mean_abs_shap"]))

    for feat in all_features:
        s_score = stat_map.get(feat, 0.0)
        shap_score = shap_dict.get(feat, 0.0)
        # weights: shap (tree-based) more importance but keep both
        combined = (shap_score * 2.0) + s_score
        combined_scores.append({"Feature": feat, "stat_score": s_score, "shap_score": shap_score, "combined_score": combined})

    combined_df = pd.DataFrame(combined_scores).sort_values("combined_score", ascending=False).reset_index(drop=True)
    return combined_df


# ---------------------------
# 7) FULL PIPELINE: prepare_and_save_model
# ---------------------------
def prepare_and_save_model(
    data_path: str = "Churn modelling.csv",
    config: Dict[str, Any] = DEFAULT_CONFIG
) -> Dict[str, Any]:
    """
    Orchestrates the entire flow: analysis -> preprocess -> SMOTE -> select -> train -> interpret -> save.

    Returns artifacts dict and writes a joblib file to config['model_output_path'].
    """
    import joblib

    # Load
    df, target_col = load_data(data_path, config.get("target_col"))
    # Drop obvious identifiers if present to reduce leakage
    drop_candidates = ["RowNumber", "CustomerId", "CustomerID", "Surname", "Name"]
    df = df.drop(columns=[c for c in drop_candidates if c in df.columns], errors='ignore')

    # Basic stats
    churn_rate = df[target_col].mean()
    print(f"Churn rate: {churn_rate:.2%}")

    # Univariate
    stat_df = univariate_analysis(df, target_col)

    # Optional plots
    if config.get("plot", False):
        visual_exploration(df, target_col)

    # Prepare X/y and split
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.get("test_size", 0.2),
        stratify=y, random_state=config.get("random_state", 42)
    )

    # Train pipeline on TRAIN ONLY (preprocessor fit on train)
    train_artifacts = train_model(X_train, y_train, config)

    # Prepare selected training and testing arrays for SHAP & final eval
    # Use helper process_and_select_func
    X_train_sel = train_artifacts["process_and_select_func"](X_train)
    X_test_sel = train_artifacts["process_and_select_func"](X_test)

    # SHAP & importances
    explainer, shap_vals, shap_df = compute_shap_and_importances(train_artifacts["model"], X_test_sel, train_artifacts["selected_features"])

    # Combined ranking
    combined_df = build_combined_ranking(stat_df, shap_df)

    # Evaluate final model on test set (simple metrics)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    model = train_artifacts["model"]
    y_pred = model.predict(X_test_sel)
    y_proba = model.predict_proba(X_test_sel)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba))
    }

    # Build feature origin map: map processed selected features to original feature (for SHAP mapping)
    # processed_feature_names -> original via split on '_' for OHE; numeric keep same
    processed_all = train_artifacts["processed_feature_names"]
    selected = train_artifacts["selected_features"]
    def origin_of(proc_name):
        if "_" in proc_name:
            return proc_name.split("_")[0]
        else:
            return proc_name

    feature_origin_map = {p: origin_of(p) for p in selected}

    # Save artifacts
    artifacts = {
        "model": model,
        "preprocessor": train_artifacts["preprocessor"],
        "selector": train_artifacts["selector"],
        "explainer": explainer,                 # shap explainer (TreeExplainer)
        "shap_sample_values": shap_vals,        # shap values for X_test_sel (can be large)
        "shap_global": shap_df,                 # DataFrame
        "statistical_results": stat_df,
        "combined_ranking": combined_df,
        "feature_origin_map": feature_origin_map,
        "processed_feature_names": processed_all,
        "selected_features": selected,
        "original_num_cols": train_artifacts["original_num_cols"],
        "original_cat_cols": train_artifacts["original_cat_cols"],
        "metrics": metrics,
        "config": config
    }

    outpath = config.get("model_output_path", DEFAULT_CONFIG["model_output_path"])
    print(f"\nSaving artifacts to '{outpath}' ...")
    joblib.dump(artifacts, outpath)
    print("Saved model artifacts.")

    # Basic printouts
    print("\nTop SHAP features:")
    print(artifacts["shap_global"].head(10).to_string(index=False))
    print("\nTop combined ranking:")
    print(artifacts["combined_ranking"].head(10).to_string(index=False))
    print("\nMetrics:")
    print(artifacts["metrics"])

    return artifacts


# ---------------------------
# CLI / Main
# ---------------------------
if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Prepare Churn Model (CatBoost) with interpretability.")
    parser.add_argument("--data", type=str, default="Churn modelling.csv", help="Path to CSV dataset")
    parser.add_argument("--out", type=str, default=DEFAULT_CONFIG['model_output_path'], help="Path to save model artifacts (.joblib)")
    parser.add_argument("--plot", action="store_true", help="Show plots during analysis")
    parser.add_argument("--target", type=str, default=DEFAULT_CONFIG['target_col'], help="Target column name (default 'Exited')")

    # ✅ Ignore Colab/Jupyter extra arguments
    args, unknown = parser.parse_known_args(sys.argv[1:])

    cfg = DEFAULT_CONFIG.copy()
    cfg["model_output_path"] = args.out
    cfg["plot"] = args.plot
    cfg["target_col"] = args.target

    artifacts = prepare_and_save_model(data_path=args.data, config=cfg)
    print("\nDone.")

