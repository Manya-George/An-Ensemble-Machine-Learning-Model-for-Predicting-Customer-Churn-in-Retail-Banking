import joblib
import numpy as np
import pandas as pd
import shap
import json
from core.config import *
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Any
import traceback

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Replicates the dynamic feature engineering logic."""
    def __init__(self, inferred_features):
        self.inferred_features = inferred_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        balance_col = next((c for c in self.inferred_features.get('financial_like', []) if 'balance' in c.lower() and c in df.columns), None)
        salary_col = next((c for c in self.inferred_features.get('financial_like', []) if 'salary' in c.lower() and c in df.columns), None)
        age_col = next((c for c in self.inferred_features.get('demographic_like', []) if 'age' in c.lower() and c in df.columns), None)
        tenure_col = next((c for c in self.inferred_features.get('relational_like', []) if 'tenure' in c.lower() and c in df.columns), None)
        
        if balance_col is not None:
            df['FE_ZeroBalance'] = (df[balance_col] == 0).astype(int)
            if salary_col is not None:
                df['FE_BalanceToSalaryRatio'] = df[balance_col] / (df[salary_col] + 1)
            
        if age_col is not None and tenure_col is not None:
            df['FE_AgeTenureInteraction'] = df[age_col] * df[tenure_col]
            
        if tenure_col is not None:
            try:
                df['FE_TenureGroup'] = pd.cut(df[tenure_col], bins=[-1, 2, 5, 10], labels=['New', 'Medium', 'Long'], right=True, include_lowest=True).astype(str)
            except Exception:
                df['FE_TenureGroup'] = 'Unknown'

        cols_to_drop = [c for c in self.inferred_features.get('id_cols', []) if c in df.columns]
        df = df.drop(columns=cols_to_drop, errors="ignore")
        
        return df

def safe_json(data):
    """Convert Python/Numpy structures into JSON-safe values."""
    def fix_value(v):
        if isinstance(v, (np.integer, np.int32, np.int64)):
            return int(v)
        if isinstance(v, (np.floating, np.float32, np.float64)):
            if np.isnan(v) or np.isinf(v):
                return 0  # or None, but 0 is safest
            return float(v)
        if isinstance(v, (np.ndarray, list, tuple)):
            return [fix_value(i) for i in v]
        if isinstance(v, dict):
            return {str(k): fix_value(val) for k, val in v.items()}
        if v is None:
            return None
        return v

    cleaned = fix_value(data)
    return json.dumps(cleaned, ensure_ascii=False)

# FIXED: Define as dict mapping canonical to possible variations
CANONICAL_COLUMNS = {
    "CustomerId": ["customerid", "customer_id", "custid"],
    "Surname": ["surname", "lastname", "last_name"],
    "CreditScore": ["creditscore", "credit_score"],
    "Geography": ["geography", "country", "location", "county", "branch", "residence"],
    "Gender": ["gender", "sex"],
    "Age": ["age"],
    "Tenure": ["tenure"],
    "Balance": ["balance", "account_balance"],
    "NumOfProducts": ["numofproducts", "num_of_products", "products", "product_count"],
    "HasCrCard": ["hascrcard", "has_cr_card", "has_credit_card", "creditcard"],
    "IsActiveMember": ["isactivemember", "is_active_member", "active", "is_active"],
    "EstimatedSalary": ["estimatedsalary", "estimated_salary", "salary"]
}

def _build_original_row_from_db(raw_row, inferred_features=None):
    """
    Accepts raw_row as dict OR pandas Series OR 1-row DataFrame.
    Normalizes into a dict containing EXACT CANONICAL_COLUMNS.
    """
    try:
        # --- Normalize raw_row into a Python dict ---
        if isinstance(raw_row, pd.DataFrame):
            if raw_row.shape[0] != 1:
                raise ValueError("raw_row DataFrame must have exactly one row")
            raw_row = raw_row.iloc[0].to_dict()

        elif isinstance(raw_row, pd.Series):
            raw_row = raw_row.to_dict()

        elif not isinstance(raw_row, dict):
            raise ValueError(f"raw_row must be dict/Series/DataFrame, got {type(raw_row)}")

        # Lowercase keys for safe lookup
        raw_lower = {str(k).lower(): v for k, v in raw_row.items()}

        # Build canonical row
        canonical = {}
        for orig_col, lower_key_list in CANONICAL_COLUMNS.items():
            selected_value = None
            for key in lower_key_list:
                if key in raw_lower:
                    selected_value = raw_lower[key]
                    break
            canonical[orig_col] = selected_value

        # Return a DataFrame with EXACT canonical columns
        return pd.DataFrame([canonical])

    except Exception as e:
        print(f"❌ Error building original row: {e}")
        traceback.print_exc()
        raise

def compute_shap_for_single_row(raw_row, bg_df=None):
    """
    Full SHAP pipeline:
    dict/Series/DataFrame → canonical row → FE → preprocess → shap → top factors
    """
    try:
        if MODEL is None or PREPROCESSOR is None or EXPLAINER is None:
            print("⚠️ SHAP skipped: model artifacts missing")
            return []

        # Step 1: canonical original DF
        df_original = _build_original_row_from_db(raw_row)

        # Step 2: Feature engineering
        if FEATURE_ENGINEER:
            try:
                df_engineered = FEATURE_ENGINEER.transform(df_original.copy())
            except Exception as e:
                print(f"⚠️ Feature engineering failed: {e}")
                df_engineered = df_original.copy()
        else:
            df_engineered = df_original.copy()

        # Step 3: Preprocessing (must remain 2D!)
        X_proc = PREPROCESSOR.transform(df_engineered)
        X_proc = np.array(X_proc).reshape(1, -1)  # safety reshape

        # Step 4: SHAP
        shap_raw = EXPLAINER.shap_values(X_proc)
        shap_vals = shap_raw[1] if isinstance(shap_raw, list) else shap_raw

        feature_names = list(PREPROCESSOR.get_feature_names_out())
        shap_values_1d = np.array(shap_vals[0]) if len(shap_vals.shape) > 1 else np.array(shap_vals)  # ensure 1D

        # Step 5: Top 3 risk factors
        top_factors = _get_top_risk_factors_from_shap(
            shap_values_1d, feature_names, df_original.iloc[0].to_dict(), top_n=3
        )
        return top_factors

    except Exception as e:
        print(f"❌ On-demand SHAP compute failed: {e}")
        traceback.print_exc()
        return []

def _get_top_risk_factors_from_shap(shap_values_row: np.ndarray,
                                   processed_feature_names: List[str],
                                   raw_data_row: Dict[str, Any],
                                   top_n: int = 3,
                                   abs_threshold: float = 0.001) -> List[Dict[str, str]]:
    """
    Map a single SHAP row (values aligned to processed_feature_names) back to human-friendly
    original feature names and return top N features pushing toward churn.
    """
    try:
        if shap_values_row is None or processed_feature_names is None:
            return []

        # Ensure consistent Python float array
        sv = np.array(shap_values_row).astype(float)
        if sv.size == 0:
            return []

        # Build dataframe for sorting
        df_sh = pd.DataFrame({
            'processed_name': processed_feature_names[:len(sv)],  # Ensure same length
            'shap_value': sv
        })

        # Compute absolute values and pick top_n by absolute SHAP
        df_sh['abs_val'] = df_sh['shap_value'].abs()
        df_sh = df_sh[df_sh['abs_val'] >= abs_threshold]  # Filter noise
        df_sh = df_sh.sort_values('abs_val', ascending=False).head(top_n)

        # Normalize raw keys for lookup (lowercase)
        raw_lower = {k.lower(): v for k, v in (raw_data_row or {}).items()}

        risk_factors = []
        for _, row in df_sh.iterrows():
            proc_name = row['processed_name']
            shap_val = float(row['shap_value'])

            # Derive an original-friendly feature label
            parts = proc_name.split('__', 1)
            if len(parts) == 2:
                prefix, feature_part = parts
            else:
                prefix, feature_part = '', proc_name

            # For one-hot encoded categorical like "Geography_France", split to base feature and value
            if prefix.startswith('cat') and '_' in feature_part:
                base_parts = feature_part.split('_', 1)
                if len(base_parts) == 2:
                    base, onehot_val = base_parts
                    original_feature = base
                    raw_value = onehot_val
                else:
                    original_feature = feature_part
                    raw_value = 'N/A'
            else:
                original_feature = feature_part
                # Try to find the matching raw value from raw_data_row (case-insensitive)
                raw_key_candidate = original_feature.replace('FE_', '').replace('num__', '').replace('cat__', '')
                
                raw_val = None
                # Try exact match first
                if raw_key_candidate.lower() in raw_lower:
                    raw_val = raw_lower[raw_key_candidate.lower()]
                else:
                    # Try common variants
                    alt_keys = [
                        raw_key_candidate,
                        raw_key_candidate.replace('_', ''),
                        raw_key_candidate.replace('_', ' ').title().replace(' ', '')
                    ]
                    for k in alt_keys:
                        if k.lower() in raw_lower:
                            raw_val = raw_lower[k.lower()]
                            break
                
                raw_value = raw_val if raw_val is not None else 'N/A'

            # Compose a human impact description
            if shap_val > 0:
                direction = 'increases'
            else:
                direction = 'decreases'

            # Friendly label
            friendly_name = original_feature.replace('FE_', '').replace('_', ' ').strip().title()

            # Build impact message
            impact = f"{friendly_name} ({raw_value}): This feature {direction} churn risk (impact: {abs(shap_val):.3f})"

            risk_factors.append({
                'factor': friendly_name,
                'value': str(raw_value),
                'impact': impact
            })

        return risk_factors[:top_n]  # Ensure we return exactly top_n
        
    except Exception as e:
        print(f"❌ Error in _get_top_risk_factors_from_shap: {e}")
        traceback.print_exc()
        return []


# ML ARTIFACTS LOADING

MODEL, PREPROCESSOR, THRESHOLD, INFERRED_FEATURES, EXPLAINER, FEATURE_ENGINEER = None, None, 0.5, None, None, None

try:
    model_artifacts = joblib.load(MODEL_PATH)
    
    # Load core components
    MODEL = model_artifacts['model']
    PREPROCESSOR = model_artifacts['fitted_preprocessor'] 
    THRESHOLD = float(model_artifacts.get('metrics', {}).get('threshold_used', 0.5))
    INFERRED_FEATURES = model_artifacts.get('inferred_features', {})
    
    # Initialize EXPLAINER (CatBoost base model required for TreeExplainer)
    catboost_model = None
    if hasattr(MODEL, 'estimators_'):
        for name, estimator in MODEL.named_estimators_.items():
            if name == 'catboost':
                catboost_model = estimator
                break
    
    if catboost_model is None:
        catboost_model = MODEL # Fallback for non-stacked model
        
    EXPLAINER = shap.TreeExplainer(catboost_model)

    # Initialize the Feature Engineer class
    FEATURE_ENGINEER = FeatureEngineer(INFERRED_FEATURES)
    
    print(f"✅ Model loaded successfully! Using optimal threshold: {THRESHOLD:.3f}")
    print(f"📊 SHAP explainer initialized with {type(catboost_model).__name__}")
    
except Exception as e:
    print(f"❌ Error loading model artifacts from {MODEL_PATH}: {e}")
    traceback.print_exc()
    # Ensure variables remain None if load fails
    MODEL, PREPROCESSOR, THRESHOLD, EXPLAINER, FEATURE_ENGINEER = None, None, 0.5, None, None