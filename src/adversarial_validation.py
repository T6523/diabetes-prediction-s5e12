from sklearn.pipeline import Pipeline
from preprocessing import get_xgb_preprocessor
from xgboost import XGBClassifier
import pandas as pd

class Validator:
    def __init__(self, df, target_name):
        self.original_df = df.copy() 
        self.target_name = target_name
        
        drop_cols = [target_name, 'diagnosed_diabetes', 'id']
        self.y_av = df[target_name]
        self.X_av = df.drop(drop_cols, axis=1, errors='ignore')
        
        self.model = Pipeline([
            ('preprocessor', get_xgb_preprocessor()),
            ('clf', XGBClassifier(
                n_estimators=50,   
                max_depth=4,       
                n_jobs=-1, 
                random_state=42,
                eval_metric='logloss'
            ))
        ])

        self.train()
        self.predict()
    
    def train(self):
        print("Training Adversarial Validator (XGB)")
        self.model.fit(self.X_av, self.y_av)
    
    def predict(self):
        self.pred = self.model.predict_proba(self.X_av)[:, 1]

    def get_filtered(self, threshold=0.75):
        mask = self.get_mask(threshold)
        dropped_count = len(self.original_df) - mask.sum()
        print(f"Adversarial Filtering: Dropped {dropped_count} rows")
        return self.original_df[mask].copy()
    
    def get_mask(self, threshold=0.75):
        mask = (self.y_av == 0) | (self.pred < threshold)
        return mask