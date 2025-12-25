import pandas as pd
import os
from sklearn.model_selection import train_test_split
from config import TARGET, NUMERIC_COLS

class DiabetesLoader:
    def __init__(self, filepath):
        self.filepath = os.path.abspath(filepath)
        self.df = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.ex = None
        self.load_and_clean()

    def load_and_clean(self):
        print(f"Loading data from {self.filepath}")
        self.df = pd.read_csv(self.filepath)
        
        if 'id' in self.df.columns:
            self.df = self.df.drop(columns=['id'])
            
        self.df = self.df.dropna()
        print(f"Data Loaded. Shape: {self.df.shape}")


    def split_data(self, split_ratio=0.8):
        target_col = TARGET
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=(1 - split_ratio), random_state=42
        )

        print(f"Train X: {self.X_train.shape}  Train y: {self.y_train.shape}")
        print(f"Val X:   {self.X_val.shape}    Val y:   {self.y_val.shape}")
        
    def get_data(self):
        self.df.drop_duplicates(inplace=True)
        self.split_data()
        
        return self.X_train, self.y_train, self.X_val, self.y_val

    def get_full(self):
        self.df.drop_duplicates(inplace=True)
        return self.df.drop(columns=[TARGET]), self.df[TARGET]

    def load_external(self, external_filepath):

        print(f"loading external data from {external_filepath}")
        self.ex = pd.read_csv(external_filepath)
        self.ex = self.ex[self.df.columns]

        if 'id' in self.ex.columns:
            self.ex = self.ex.drop(columns=['id'])

        self.ex['is_external'] = 1
        self.df['is_external'] = 0

        for col in NUMERIC_COLS:
            min_val = self.df[col].min()
            max_val = self.df[col].max()
        
            self.ex = self.ex[
                (self.ex[col] >= min_val) & 
                (self.ex[col] <= max_val)
            ]

        self.df = pd.concat([self.df,self.ex])



class TestLoader:
    def __init__(self, filepath):
        self.filepath = os.path.abspath(filepath)
        self.df = None
        self.id = None
    
    def load_data(self):
        print(f"Loading data from {self.filepath}")
        self.df = pd.read_csv(self.filepath)
        self.id = self.df.id

    def get_data(self):
        self.load_data()
        print(f"Test X: {self.df.shape}")
        return self.df.drop(columns=['id']), self.id