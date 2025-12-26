import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import time
from sklearn.base import clone

def evaluate_model(model, X, y, n_splits=5, random_state=42):
    """
    Trustworthy CV Strategy:
    1. Splits data based on 'is_external' column.
    2. CV Loop on Synthetic Data Only.
    3. Training Data = Synthetic Fold + ALL External Data.
    4. DROPS 'is_external' (and 'id') before training so the model 
       doesn't learn to distinguish sources.
    """
    start_time = time.time()
    
    #  Validate
    if 'is_external' not in X.columns:
        raise ValueError("X must contain 'is_external' column for this validation strategy.")
    
    # Separate Synthetic and External
    mask_syn = X['is_external'] == 0
    mask_ext = X['is_external'] == 1
    
    X_syn = X[mask_syn]
    y_syn = y[mask_syn]
    
    X_ext = X[mask_ext]
    y_ext = y[mask_ext]
    
    # Stratified K-Fold on Synthetic 
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    
    print(f"Starting CV on {type(model).__name__}...")
    print(f"Data Split: {len(X_syn)} Synthetic | {len(X_ext)} External")
    
    cols_to_drop = ['is_external', 'id']
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_syn, y_syn)):
        # Validation Set 
        X_val = X_syn.iloc[val_idx].copy()
        y_val = y_syn.iloc[val_idx]
        
        # Training Set (Synthetic Fold + All External)
        X_train_fold = X_syn.iloc[train_idx]
        y_train_fold = y_syn.iloc[train_idx]
        X_train_final = pd.concat([X_train_fold, X_ext], axis=0)
        y_train_final = pd.concat([y_train_fold, y_ext], axis=0)
        
        # CLEAN FEATURES (Drop 'is_external')
        X_train_clean = X_train_final.drop(columns=cols_to_drop, errors='ignore')
        X_val_clean = X_val.drop(columns=cols_to_drop, errors='ignore')
        
        # Train 
        # Clone model to ensure fresh start
        curr_model = clone(model)
        curr_model.fit(X_train_clean, y_train_final)
        
        # Predict & Score
        if hasattr(curr_model, "predict_proba"):
            val_probs = curr_model.predict_proba(X_val_clean)[:, 1]
        else:
            val_probs = curr_model.predict(X_val_clean)
            
        score = roc_auc_score(y_val, val_probs)
        scores.append(score)
        print(f"  Fold {fold+1}: {score:.4f}")
        
    # Aggregate
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    elapsed = time.time() - start_time
    
    print(f"--> Result: {mean_score:.5f} ± {std_score:.5f} (Time: {elapsed:.1f}s)")
    
    return {'mean': mean_score, 'std': std_score, 'scores': scores}