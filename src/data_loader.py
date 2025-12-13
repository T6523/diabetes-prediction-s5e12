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

    def load_and_clean(self):
        print(f"Loading data from {self.filepath}")
        self.df = pd.read_csv(self.filepath)
        
        if 'id' in self.df.columns:
            self.df = self.df.drop(columns=['id'])
            
        self.df = self.df.dropna()
        print(f"Data Loaded. Shape: {self.df.shape}")


    def split_data(self, split_ratio=0.8):
        target_col = 'diagnosed_diabetes'
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=(1 - split_ratio), random_state=42
        )
        
    def get_data(self):
        self.load_and_clean()
        self.split_data()
    
        print(f"Train X: {self.X_train.shape}  Train y: {self.y_train.shape}")
        print(f"Val X:   {self.X_val.shape}    Val y:   {self.y_val.shape}")
        
        return self.X_train, self.y_train, self.X_val, self.y_val