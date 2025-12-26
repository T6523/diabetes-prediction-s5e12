import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import time
from sklearn.base import clone

def evaluate_model(model, X, y, n_splits=5, random_state=42):
    start_time = time.time()
    
    if 'is_external' not in X.columns:
        raise ValueError("X must contain 'is_external' column.")
    
    mask_syn = X['is_external'] == 0
    mask_ext = X['is_external'] == 1
    
    cols_to_drop = ['is_external', 'id']
    X_clean = X.drop(columns=cols_to_drop, errors='ignore')
    
    X_syn = X_clean[mask_syn].values
    y_syn = y[mask_syn].values
    X_ext = X_clean[mask_ext].values
    y_ext = y[mask_ext].values
    
    feature_names = X_clean.columns.tolist()
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    
    print(f"Starting CV on {type(model).__name__}...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_syn, y_syn)):
        X_val = X_syn[val_idx]
        y_val = y_syn[val_idx]
        
        X_train_fold = X_syn[train_idx]
        y_train_fold = y_syn[train_idx]
        
        X_train_final = np.vstack([X_train_fold, X_ext])
        y_train_final = np.concatenate([y_train_fold, y_ext])
        
        X_train_df = pd.DataFrame(X_train_final, columns=feature_names).infer_objects()
        X_val_df = pd.DataFrame(X_val, columns=feature_names).infer_objects()
        
        curr_model = clone(model)
        curr_model.fit(X_train_df, y_train_final)
        
        if hasattr(curr_model, "predict_proba"):
            val_probs = curr_model.predict_proba(X_val_df)[:, 1]
        else:
            val_probs = curr_model.predict(X_val_df)
            
        score = roc_auc_score(y_val.astype(int), val_probs)
        scores.append(score)
        print(f"Fold {fold+1}: {score:.4f}")
        
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    elapsed = time.time() - start_time
    
    print(f"Result: {mean_score:.5f} ± {std_score:.5f} (Time: {elapsed:.1f}s)")
    
    return {'mean': mean_score, 'std': std_score, 'scores': scores}