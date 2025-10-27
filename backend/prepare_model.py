#prepare_model.py

"""
Script to prepare and save the churn prediction model for production use.
Run this after training your model in the Jupyter notebook.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

def prepare_and_save_model(data_path='Churn modelling.csv', model_output_path='catboost_churn_model.joblib'):
    """
    Load data, train model, and save for production use
    
    Args:
        data_path: Path to training CSV file
        model_output_path: Path to save the model artifacts
    """
    
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Configuration
    TARGET_COL = "Exited"
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # Split features and target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Identify column types
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"Numeric columns: {num_cols}")
    print(f"Categorical columns: {cat_cols}")
    
    # Build preprocessing pipeline
    print("\nBuilding preprocessing pipeline...")
    
    import sklearn
    from packaging import version
    if version.parse(sklearn.__version__) >= version.parse("1.2"):
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", ohe)
        ]), cat_cols)
    ])
    
    # Fit preprocessor
    print("Fitting preprocessor...")
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    print(f"Preprocessed shape: {X_train_proc.shape}")
    
    # Feature selection
    print("\nPerforming feature selection...")
    sel_est = LogisticRegression(penalty="l1", solver="saga", max_iter=5000, random_state=RANDOM_STATE)
    sel_est.fit(X_train_proc, y_train)
    
    selector = SelectFromModel(sel_est, prefit=True, threshold="median")
    X_train_sel = selector.transform(X_train_proc)
    X_test_sel = selector.transform(X_test_proc)
    
    print(f"Selected features: {X_train_sel.shape[1]}")
    
    # Train final CatBoost model with optimal parameters
    print("\nTraining CatBoost model...")
    
    best_params = {
        'iterations': 1275,
        'depth': 9,
        'learning_rate': 0.03418620644574627,
        'l2_leaf_reg': 0.3784291068462627,
        'rsm': 0.768358600247406,
        'bagging_temperature': 0.786489153878744,
        'random_seed': RANDOM_STATE,
        'verbose': 100
    }
    
    final_model = CatBoostClassifier(**best_params)
    final_model.fit(
        X_train_sel, y_train,
        eval_set=(X_test_sel, y_test),
        early_stopping_rounds=50
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    y_pred = final_model.predict(X_test_sel)
    y_proba = final_model.predict_proba(X_test_sel)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    # Save model and artifacts
    print(f"\nSaving model to {model_output_path}...")
    
    model_artifacts = {
        "model": final_model,
        "preprocessor": preprocessor,
        "selector": selector,
        "best_params": best_params,
        "metrics": metrics,
        "feature_names": {
            'numeric': num_cols,
            'categorical': cat_cols
        }
    }
    
    joblib.dump(model_artifacts, model_output_path)
    
    print(f"Model saved successfully to {model_output_path}")
    print("\nModel is ready for production use!")
    
    return model_artifacts

if __name__ == "__main__":
    # Prepare and save the model
    prepare_and_save_model()
    
    # Test loading the model
    print("\n" + "="*50)
    print("Testing model loading...")
    loaded = joblib.load('catboost_churn_model.joblib')
    print("Model loaded successfully!")
    print(f"Model type: {type(loaded['model'])}")
    print(f"Model metrics: {loaded['metrics']}")