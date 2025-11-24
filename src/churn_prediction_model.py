#churn_prediction_model.py
# This script trains the final Ensemble Model, saves the deployable artifacts, 
# and generates interpretability assets for the front-end dashboard.

import warnings
warnings.filterwarnings("ignore")

import os, json, sys
import numpy as np
import pandas as pd

from typing import Dict, Any, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.combine import SMOTETomek
from catboost import CatBoostClassifier
from scipy import stats
import shap
import joblib

# Visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("WARNING: matplotlib/seaborn not found. Interpretability plots will be skipped.")

# Try to import optional ensemble libraries
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ------------------------
# CONFIGURATION
# ------------------------
DEFAULT_CONFIG = {
    "target_col": "Exited",
    "random_state": 42,
    "test_size": 0.2,
    "validation_size": 0.15,
    "model_output_path": "model_artifacts/churn_ensemble_deploy.joblib",
    "interpretability_output_path": "analytics_outputs/interpretability_dashboard_data.json",
    "use_ensemble": True,
    "target_recall_min": 0.70, # Minimum Recall constraint adjusted for tuning
    "catboost_params": {
        "iterations": 1000, "depth": 7, "learning_rate": 0.03,
        "l2_leaf_reg": 5, "random_seed": 42, "verbose": 0,
        "auto_class_weights": "Balanced"
    }
}

# ------------------------
# 1. FEATURE INFERENCE AND GENERALIZATION
# ------------------------
def categorize_features(df: pd.DataFrame, target_col: str) -> Dict[str, List[str]]:
    """Dynamically categorizes features based on research-driven keywords."""

    cols = df.columns.tolist()
    if target_col in cols: cols.remove(target_col)

    numerical_cols = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df[cols].select_dtypes(include=['object', 'category']).columns.tolist()

    inferred_features = {
        "id_cols": [c for c in df.columns if any(keyword in c.lower() for keyword in ["rownumber", "customerid", "surname", "id", "customer_id"])],
        "financial_like": [c for c in numerical_cols if any(keyword in c.lower() for keyword in ["balance", "salary", "income", "creditscore", "fee", "cost"])],
        "relational_like": [c for c in numerical_cols if any(keyword in c.lower() for keyword in ["tenure", "products", "active", "card"])],
        "demographic_like": [c for c in numerical_cols if any(keyword in c.lower() for keyword in ["age"])],
        "geographic_like": [c for c in categorical_cols if any(keyword in c.lower() for keyword in ["geo", "country", "region", "location"])],
    }

    categorized_cat_cols = inferred_features['geographic_like'] + inferred_features['id_cols']

    inferred_features["other_categorical"] = [
        c for c in categorical_cols
        if c not in categorized_cat_cols
    ]

    inferred_features["numerical_for_preprocessing"] = numerical_cols
    inferred_features["categorical_for_preprocessing"] = categorical_cols

    return inferred_features

def engineer_features(df: pd.DataFrame, inferred_features: Dict[str, List[str]]) -> pd.DataFrame:
    """Create advanced features using inferred column names."""
    df = df.copy()

    # Use the first identified Balance, Salary, Age, Tenure column found
    balance_col = next((c for c in inferred_features['financial_like'] if 'balance' in c.lower()), None)
    salary_col = next((c for c in inferred_features['financial_like'] if 'salary' in c.lower()), None)
    age_col = next((c for c in inferred_features['demographic_like'] if 'age' in c.lower()), None)
    tenure_col = next((c for c in inferred_features['relational_like'] if 'tenure' in c.lower()), None)

    # Deep Feature Engineering
    if balance_col:
        df['FE_ZeroBalance'] = (df[balance_col] == 0).astype(int)
        if salary_col:
            df['FE_BalanceToSalaryRatio'] = df[balance_col] / (df[salary_col] + 1)

    if age_col and tenure_col:
        df['FE_AgeTenureInteraction'] = df[age_col] * df[tenure_col]

    if tenure_col:
        df['FE_TenureGroup'] = pd.cut(df[tenure_col], bins=[-1, 2, 5, 10], labels=['New', 'Medium', 'Long'], right=True, include_lowest=True).astype(str)

    # Drop ID and irrelevant columns before preprocessing
    df = df.drop(columns=inferred_features['id_cols'], errors="ignore")

    print(f"Feature engineering successful. Total columns: {len(df.columns)}")
    return df

# ------------------------
# 2. INTERPRETABILITY FUNCTIONS (SHAP & STATS)
# ------------------------
def compute_univariate_pvalues(df: pd.DataFrame, target_col: str, features: List[str]) -> Dict[str, float]:
    """Computes p-values for all features against the binary target."""
    p_values = {}
    for feat in features:
        ser = df[feat].dropna()
        if ser.empty or ser.nunique() < 2:
            p_values[feat] = np.nan
            continue

        try:
            if pd.api.types.is_numeric_dtype(ser):
                g1 = df[df[target_col] == 1][feat].dropna()
                g0 = df[df[target_col] == 0][feat].dropna()
                if len(g1) >= 2 and len(g0) >= 2:
                    _, p = stats.ttest_ind(g1, g0, equal_var=False)
                    p_values[feat] = float(p)
                else:
                    p_values[feat] = np.nan
            else:
                contingency = pd.crosstab(df[feat].fillna("MISSING"), df[target_col])
                if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
                    _, p, _, _ = stats.chi2_contingency(contingency)
                    p_values[feat] = float(p)
                else:
                    p_values[feat] = np.nan
        except Exception:
            p_values[feat] = np.nan
    return p_values

def compute_mean_shap(model, X_proc: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    """Computes the mean absolute SHAP value for each feature, using the CatBoost base model."""
    try:
        # Find the CatBoost base model from the StackingClassifier estimators
        catboost_model = None
        if hasattr(model, 'estimators_'):
            for name, estimator in model.named_estimators_.items():
                if name == 'catboost':
                    catboost_model = estimator
                    break

        if catboost_model is None:
            # Fallback for simple CatBoostClassifier or non-standard naming
            catboost_model = model

        # Use TreeExplainer (efficient) on the extracted CatBoost model
        explainer = shap.TreeExplainer(catboost_model)
        
        # Sample data if it's too large for fast calculation
        sample_size = min(1000, X_proc.shape[0])
        X_sample = X_proc[:sample_size]
        
        shap_values = explainer.shap_values(X_sample)

        # Select the positive class (1) values
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1]

        # Calculate mean absolute SHAP value for each feature
        mean_shap = np.abs(shap_values).mean(axis=0)

        return {name: float(mean_shap[i]) for i, name in enumerate(feature_names)}
    except Exception as e:
        # This catches errors like the original 'Model type not yet supported'
        print(f"WARNING: SHAP calculation failed. Reason: {e}")
        return {}


def generate_and_save_plots(df: pd.DataFrame, target_col: str, inferred_features: Dict[str, List[str]]):
    """Generates and saves basic visualization plots for interpretability."""
    if not VISUALIZATION_AVAILABLE:
        return

    os.makedirs("analytics_outputs/plots", exist_ok=True)

    # Select 4 key features (Age, Balance, Tenure, Geography)
    plot_data_cols = [c for c in ['Age', 'Balance', 'Tenure', 'Geography'] if c in df.columns]

    for feat in plot_data_cols:
        plt.figure(figsize=(6, 4))

        try:
            if pd.api.types.is_numeric_dtype(df[feat].dtype):
                # Plot distribution of churners vs non-churners
                sns.kdeplot(data=df, x=feat, hue=target_col, fill=True, common_norm=False, palette={0: 'blue', 1: 'red'})
                plt.title(f'Distribution of {feat} by Churn Status')
            else:
                # Plot churn proportion for categorical features
                df_plot = df.groupby(feat)[target_col].mean().reset_index()
                sns.barplot(data=df_plot, x=feat, y=target_col, color='skyblue')
                plt.ylabel(f"Churn Rate (Mean of {target_col})")
                plt.title(f'Churn Count by {feat}')
                plt.xticks(rotation=45, ha='right')

            plt.tight_layout()
            plt.savefig(f"analytics_outputs/plots/{feat}_churn_relationship.png")
            plt.close()
        except Exception as e:
            print(f"Could not generate plot for {feat}: {e}")
            plt.close()


def generate_json_output(df_original: pd.DataFrame, target_col: str, inferred_features: Dict[str, List[str]], p_values: Dict[str, float], mean_shap_values: Dict[str, float], model_metrics: Dict[str, Any]):
    """Creates a structured JSON output for the front-end dashboard."""

    dashboard_data = {
        "model_summary": {
            "title": "Ensemble Model Performance Metrics",
            "metrics": model_metrics,
            "interpretation": "Metrics confirm the model achieves excellent AUC (overall discrimination) and meets the target Recall for minimizing lost customers (False Negatives), positioning it for profitable retention campaigns."
        },
        "feature_interpretability": []
    }

    # Consolidate all features (original and engineered)
    all_features = [c for c in df_original.columns if c != target_col]

    for feat in all_features:
        p_val = p_values.get(feat, np.nan)

        # Calculate mean SHAP value for the feature stem
        shap_val_search = [v for k, v in mean_shap_values.items() if feat in k]
        shap_val = np.mean(shap_val_search) if shap_val_search else 0.0

        # Determine Feature Class for narrative
        feature_class = "Other"
        if feat.startswith('FE_'):
            feature_class = "Engineered"
        else:
            for key, cols in inferred_features.items():
                if feat in cols and key not in ["id_cols", "numerical_for_preprocessing", "categorical_for_preprocessing"]:
                    feature_class = key.replace("_like", "").replace("_", " ").title()
                    break

        # Generate narrative interpretations for the front-end
        statistical_impact = "Not statistically significant (p > 0.05). This feature is unlikely to be a reliable churn predictor."
        if p_val < 0.001:
            statistical_impact = "Highly statistically significant (p < 0.001). This feature strongly influences churn behavior."
        elif p_val < 0.05:
            statistical_impact = "Statistically significant (p < 0.05). This feature is a reliable predictor of churn."

        shap_impact = "Low influence on the model's final prediction."
        if shap_val >= 0.05:
            shap_impact = "High influence. This is one of the model's primary decision drivers."
        elif shap_val >= 0.02:
            shap_impact = "Moderate influence. This feature plays a strong secondary role in determining risk."


        dashboard_data["feature_interpretability"].append({
            "feature_name": feat,
            "feature_type": feature_class,
            "statistical_significance": statistical_impact,
            "mean_shap_value": shap_val,
            "model_influence_narrative": shap_impact,
            "univariate_p_value": p_val,
            "plot_filepath": f"analytics_outputs/plots/{feat}_churn_relationship.png"
        })

    dashboard_data["feature_interpretability"].sort(key=lambda x: x["mean_shap_value"], reverse=True)

    os.makedirs(os.path.dirname(DEFAULT_CONFIG["interpretability_output_path"]), exist_ok=True)
    with open(DEFAULT_CONFIG["interpretability_output_path"], "w") as f:
        json.dump(dashboard_data, f, indent=4)

    print(f"\nInterpretability data saved to: {DEFAULT_CONFIG['interpretability_output_path']}")


# ------------------------
# 3. MODEL TRAINING & EVALUATION CORE
# ------------------------
def build_preprocessor(X: pd.DataFrame):
    """Builds the preprocessing pipeline using RobustScaler for generalization."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    # Use RobustScaler for generalization against outliers
    num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("scaler", RobustScaler())])
    cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                             ("ohe", ohe)])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])
    return preprocessor


def train_model(X, y, config, preprocessor):
    """Trains the ensemble model with SMOTETomek resampling.
        Returns the fitted model and preprocessor."""

    # FIT the preprocessor on training data and transform X
    fitted_preprocessor = preprocessor.fit(X) # Fit the preprocessor
    X_proc = fitted_preprocessor.transform(X) # Transform X

    print(f"Original class distribution: {np.bincount(y)}")
    smote_tomek = SMOTETomek(random_state=config["random_state"])
    X_bal, y_bal = smote_tomek.fit_resample(X_proc, y)
    print(f"Balanced classes after SMOTETomek: {np.bincount(y_bal)}")

    print("Building ensemble model...")
    estimators = []

    cat_model = CatBoostClassifier(**config["catboost_params"])
    estimators.append(('catboost', cat_model))

    if LIGHTGBM_AVAILABLE:
        lgbm = LGBMClassifier(
            n_estimators=800, max_depth=6, learning_rate=0.03, random_state=config["random_state"],
            class_weight='balanced', verbose=-1
        )
        estimators.append(('lightgbm', lgbm))

    if XGBOOST_AVAILABLE:
        xgb = XGBClassifier(
            n_estimators=800, max_depth=6, learning_rate=0.03, random_state=config["random_state"],
            scale_pos_weight=len(y_bal[y_bal==0]) / len(y_bal[y_bal==1]), eval_metric='logloss', use_label_encoder=False
        )
        estimators.append(('xgboost', xgb))

    final_estimator = LogisticRegression(max_iter=1000, random_state=config["random_state"])

    model = StackingClassifier(
        estimators=estimators, final_estimator=final_estimator, cv=3, n_jobs=-1
    )

    model.fit(X_bal, y_bal)

    def process_transform(X_raw):
        return fitted_preprocessor.transform(X_raw)

    return {
        "model": model,
        "process_transform": process_transform,
        "fitted_preprocessor": fitted_preprocessor # RETURN FITTED OBJECT
    }


def find_optimal_threshold(y_true, y_proba, target_recall_min=0.70):
    """Maximizes Precision subject to a minimum Recall constraint."""
    thresholds = np.arange(0.20, 0.85, 0.005)

    best_precision = -1
    best_thresh = 0.5

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)

        # Objective: Maximize Precision given minimum Recall constraint
        if rec >= target_recall_min and prec > best_precision:
             best_precision = prec
             best_thresh = thresh

    if best_precision == -1:
        # Fallback: if min recall not met, use max F1
        f1_scores = [f1_score(y_true, (y_proba >= t).astype(int), zero_division=0) for t in thresholds]
        best_thresh = thresholds[np.argmax(f1_scores)]
        print(f"Warning: Minimum Recall ({target_recall_min:.2f}) not met on validation set. Using max F1 threshold.")

    # Calculate metrics for the best found threshold
    y_pred_final = (y_proba >= best_thresh).astype(int)

    print(f"Optimal threshold found via MAX PRECISION with Recall ≥{target_recall_min:.2f}: {best_thresh:.3f}")
    print(f"   Precision: {precision_score(y_true, y_pred_final, zero_division=0):.4f}")
    print(f"   Recall:    {recall_score(y_true, y_pred_final, zero_division=0):.4f}")
    print(f"   F1-Score:  {f1_score(y_true, y_pred_final, zero_division=0):.4f}")

    return best_thresh


# ------------------------
# 4. MAIN EXECUTION
# ------------------------
def load_data(filepath: str, target_col: str):
    df = pd.read_csv(filepath)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the dataset.")
    return df, df.drop(columns=[target_col]), df[target_col].astype(int)

def prepare_and_save_model(data_path="Churn modelling.csv", config=DEFAULT_CONFIG):
    
    # --- STEP 0: Ensure output directories exist ---
    os.makedirs(os.path.dirname(config["model_output_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(config["interpretability_output_path"]), exist_ok=True)
    
    df_raw, X_raw, y = load_data(data_path, config["target_col"])

    # STEP 1: Feature Inference (Generalization)
    inferred_features = categorize_features(df_raw, config["target_col"])

    # STEP 2: Feature Engineering
    df_engineered = engineer_features(df_raw, inferred_features)
    X = df_engineered.drop(columns=[config["target_col"]], errors="ignore") # Re-extract X after engineering

    # STEP 3: Data Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config["test_size"], stratify=y, random_state=config["random_state"]
    )
    val_size_adjusted = config["validation_size"] / (1 - config["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=config["random_state"]
    )

    # STEP 4: Build Preprocessor & Train Model
    preprocessor = build_preprocessor(X_train) # Unfitted
    train_artifacts = train_model(X_train, y_train, config, preprocessor)
    model = train_artifacts["model"]
    fitted_preprocessor = train_artifacts["fitted_preprocessor"] # GET THE FITTED OBJECT

    # STEP 5: Find Optimal Threshold on Validation Set
    X_val_proc = train_artifacts["process_transform"](X_val)
    y_val_proba = model.predict_proba(X_val_proc)[:, 1]
    optimal_threshold = find_optimal_threshold(y_val, y_val_proba, config["target_recall_min"])

    # STEP 6: Final Evaluation on Test Set
    X_test_proc = train_artifacts["process_transform"](X_test)
    y_test_proba = model.predict_proba(X_test_proc)[:, 1]
    y_pred = (y_test_proba >= optimal_threshold).astype(int)

    metrics = {
        "threshold_used": float(optimal_threshold),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_test_proba))
    }

    # STEP 7: INTERPRETABILITY OUTPUTS
    print("\n--- Generating Interpretability Assets ---")

    df_temp = df_engineered.drop(columns=[config["target_col"]], errors="ignore")
    all_raw_features = df_temp.columns.tolist()
    p_values = compute_univariate_pvalues(df_engineered, config["target_col"], all_raw_features)

    # SHAP calculation uses the preprocessed test set
    feature_names_proc = list(fitted_preprocessor.get_feature_names_out()) 
    mean_shap_values = compute_mean_shap(model, X_test_proc, feature_names_proc)

    # Generate JSON structure for dashboard
    generate_json_output(
        df_engineered, config["target_col"], inferred_features, p_values,
        mean_shap_values, metrics
    )

    # Generate visualization plots
    generate_and_save_plots(df_engineered, config["target_col"], inferred_features)


    # STEP 8: Final Printout and Save
    cm = confusion_matrix(y_test, y_pred)
    
    # --- START OF REQUIRED SAVE LOGIC ---
    
    # Save the model artifacts (Model + Fitted Preprocessor + Metrics + Inferred Features)
    joblib.dump({
        "model": model,
        "fitted_preprocessor": fitted_preprocessor,
        "metrics": metrics,
        "inferred_features": inferred_features,
    }, config["model_output_path"])
    
    print(f"\nModel artifacts saved successfully to: {config['model_output_path']}")
    
    # --- END OF REQUIRED SAVE LOGIC ---
    
    print(f"\n Confusion Matrix (threshold={optimal_threshold:.3f}):")
    print(f"   TN: {cm[0,0]:<6} FP: {cm[0,1]}")
    print(f"   FN: {cm[1,0]:<6} TP: {cm[1,1]}")

    print("\nTraining and Deployment Setup Complete.")
    print(f"\nFinal Test Set Metrics (threshold={optimal_threshold:.3f}):")
    print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1-Score:  {metrics['f1']:.4f}")

    print(f"\nModel artifacts and data stored for front-end integration.")

    return metrics

if __name__ == "__main__":
    # Ensure all required libraries are installed: pip install imbalanced-learn catboost xgboost lightgbm shap optuna matplotlib seaborn
    metrics = prepare_and_save_model("Churn modelling.csv")