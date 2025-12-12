import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class DiabetesLoader:
    def __init__(self, filepath):
        self.filepath = os.path.abspath(filepath)
        self.df = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.scaler = MinMaxScaler() 

    def load_and_clean(self):
        print(f"Loading data from {self.filepath}")
        self.df = pd.read_csv(self.filepath)
        
        # Drop ID
        if 'id' in self.df.columns:
            self.df = self.df.drop(columns=['id'])
            
        # Drop NaNs
        self.df = self.df.dropna()
        print(f"Data Loaded. Shape: {self.df.shape}")

    def encode_categoricals(self):
        # Identify string columns
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            print(f"Encoding: {list(cat_cols)}")
            self.df = pd.get_dummies(self.df, columns=cat_cols, dtype=int)
        
    def normalize(self):
        target_col = 'diagnosed_diabetes'
        
        # Separate features and target
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]
        
        # Fit and Transform using Sklearn
        X_scaled = self.scaler.fit_transform(X)
        
        # Reconstruct DataFrame (to keep it clean for the next step)
        self.df = pd.concat([
            pd.DataFrame(X_scaled, columns=X.columns),
            y.reset_index(drop=True)
        ], axis=1)
        
        print("Normalization (MinMax) complete.")

    def split_data(self, split_ratio=0.8):
        target_col = 'diagnosed_diabetes'
        X = self.df.drop(columns=[target_col]).values
        y = self.df[target_col].values
        
        # Use Scikit-Learn's splitter (Handles shuffling automatically)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=(1 - split_ratio), random_state=42
        )
        
        # Reshape y to (N, 1) for the Neural Network
        self.y_train = self.y_train.reshape(-1, 1)
        self.y_val = self.y_val.reshape(-1, 1)

    def get_data(self):
        self.load_and_clean()
        self.encode_categoricals()
        self.normalize()
        self.split_data()
    
        print(f"Train X: {self.X_train.shape} | Train y: {self.y_train.shape}")
        print(f"Val X:   {self.X_val.shape}   | Val y:   {self.y_val.shape}")
        
        return self.X_train, self.y_train, self.X_val, self.y_val