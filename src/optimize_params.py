import optuna
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

def xgb_objective(trial, ratio, X, y):
    params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 200, 600),
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 6),
            'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.1, log=True),
            'min_child_weight': trial.suggest_int('xgb_min_child', 1, 5),
            'gamma': trial.suggest_float('xgb_gamma', 0, 0.5),
            'subsample': trial.suggest_float('xgb_subsample', 0.5,1),
            'scale_pos_weight': ratio,  
            'n_jobs': -1,
            'random_state': 42,
            'eval_metric': 'auc',
            'verbosity': 0
    }
    model = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=1)
    
    return scores.mean()

def lgbm_objective(trial, ratio, X, y):
    params = {
        'n_estimators': trial.suggest_int('lgbm_n_estimators', 200, 600),
        'learning_rate': trial.suggest_float('lgbm_lr', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('lgbm_num_leaves', 20, 40),
        'scale_pos_weight': ratio,
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1
    }
    model = LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=1)
    
    return scores.mean()

def cat_objective(trial, ratio, X, y):
    params = {
            'iterations': trial.suggest_int('cat_iterations', 200, 800),
            'depth': trial.suggest_int('cat_depth', 4, 8),
            'learning_rate': trial.suggest_float('cat_lr', 0.01, 0.1, log=True),
            'l2_leaf_reg': trial.suggest_int('cat_l2', 1, 10),
            'scale_pos_weight': ratio,
            'verbose': 0,
            'random_state': 42,
            'allow_writing_files': False
        }
    model = CatBoostClassifier(**params)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=1)
    
    return scores.mean()