from xgboost import XGBClassifier

class Validator:
    def __init__(self, df, target_name):
        self.original_df = df.copy() 
        self.target_name = target_name
        self.model = XGBClassifier(n_jobs=-1, random_state=42)
        
        self.y_av = df[target_name]
        self.X_av = df.drop([target_name, 'diagnosed_diabetes'], axis=1, errors='ignore')
        
        self.train()
        self.predict()
    
    def train(self):
        self.model.fit(self.X_av, self.y_av)
    
    def predict(self):
        self.pred = self.model.predict_proba(self.X_av)[:, 1]

    def get_filtered(self, threshold=0.75):
        # Apply mask to the ORIGINAL dataframe
        mask = self.get_mask(threshold)
        print(f"Filtering: Dropped {len(self.original_df) - mask.sum()} rows")
        return self.original_df[mask].copy()
    
    def get_mask(self, threshold=0.75):
        mask = (self.y_av == 0) | (self.pred < threshold)
        return mask
        
        
        

