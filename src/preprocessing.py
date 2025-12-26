from config import NUMERIC_COLS, NOMINAL_COLS, ORDINAL_COLS, BOOL_COLS, ORDINAL_CATEGORIES
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from itertools import combinations

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        eps = 1e-6

        # cardio
        X['Pulse_Pressure'] = X['systolic_bp'] - X['diastolic_bp']
        X['MAP'] = (X['systolic_bp'] + (2 * X['diastolic_bp'])) / 3
        X['Rate_Pressure_Product'] = X['systolic_bp'] * X['heart_rate']
        X['BP_Index'] = X['systolic_bp'] / (X['diastolic_bp'] + eps)
        X['Cardio_Stress'] = X['diastolic_bp'] * X['heart_rate']

        # lipid
        X['Non_HDL'] = X['cholesterol_total'] - X['hdl_cholesterol']
        X['Trig_HDL_Ratio'] = X['triglycerides'] / (X['hdl_cholesterol'] + eps)
        X['LDL_HDL_Ratio'] = X['ldl_cholesterol'] / (X['hdl_cholesterol'] + eps)
        X['Cholesterol_Ratio'] = X['cholesterol_total'] / (X['hdl_cholesterol'] + eps)
        X['Remnant_Cholesterol'] = X['cholesterol_total'] - X['hdl_cholesterol'] - X['ldl_cholesterol']
        X['Lipid_Accumulation'] = X['triglycerides'] * X['ldl_cholesterol']
        X['Log_Triglycerides'] = np.log1p(X['triglycerides'])

        # Body Composition
        X['Visceral_Fat_Proxy'] = X['bmi'] * X['waist_to_hip_ratio']
        X['BS_Index'] = X['waist_to_hip_ratio'] / (X['bmi'] + eps)
        X['Metabolic_Syndrome_Score'] = X['bmi'] + X['waist_to_hip_ratio'] + X['systolic_bp']

        # Lifestyle
        X['Sedentary_Load'] = X['screen_time_hours_per_day'] / (X['physical_activity_minutes_per_week'] + eps)
        X['Diet_Activity_Score'] = X['diet_score'] * X['physical_activity_minutes_per_week']
        X['Unhealthy_Combo'] = X['alcohol_consumption_per_week'] * X['screen_time_hours_per_day']
        X['Relative_Activity'] = X['physical_activity_minutes_per_week'] / (X['bmi'] + eps)
        
        # Activity Efficiency 
        X['Activity_Efficiency'] = X['heart_rate'] / (X['physical_activity_minutes_per_week'] + eps)
        X['Alcohol_BMI_Risk'] = X['alcohol_consumption_per_week'] * X['bmi']
        X['Screen_BMI_Interaction'] = X['screen_time_hours_per_day'] * X['bmi']

        #  Sleep & Stress 
        X['Sleep_Stress_Proxy'] = X['heart_rate'] / (X['sleep_hours_per_day'] + eps)
        X['Sleep_Deprivation_Impact'] = X['bmi'] / (X['sleep_hours_per_day'] + eps)
        X['Morning_Risk'] = X['sleep_hours_per_day'] * X['systolic_bp']

        # Age
        X['Age_BMI_Risk'] = X['age'] * X['bmi']
        X['Age_BP_Risk'] = X['age'] * X['systolic_bp']
        X['Age_WHR_Risk'] = X['age'] * X['waist_to_hip_ratio']
        X['Vascular_Aging'] = X['age'] * X['Pulse_Pressure']
        X['Chronic_Metabolic_Load'] = X['age'] * X['triglycerides']

        return X

    def set_output(self, *, transform = None):
        return self


def get_preprocessor():

    numeric_pipe = Pipeline([
        # ('feature engineer', FeatureEngineer()),
        ('imputer', SimpleImputer(strategy='median')), 
    ])
    
    ordinal_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('encoder', OrdinalEncoder(categories=ORDINAL_CATEGORIES))
    ])
    
    nominal_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('encoder', OneHotEncoder(sparse_output=False))
    ])
    
    bool_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('encoder', OneHotEncoder(drop='first', sparse_output=False)) 
    ])

    preprocessor = ColumnTransformer(
        transformers=[
        ('num', numeric_pipe, NUMERIC_COLS),
        ('ord', ordinal_pipe, ORDINAL_COLS),
        ('nom', nominal_pipe, NOMINAL_COLS),
        ('bool', bool_pipe, BOOL_COLS)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform='pandas')

    return preprocessor

def get_top_k_features(base_model,X,y, k=10):
    k = min(k, X.shape[1])
    base_model.fit(X,y)
    feature_names = X.columns.tolist()
    importances = pd.Series(base_model.feature_importances_, index=feature_names)
    return importances.nlargest(k).index.tolist()

def is_numeric(df,col):
    return col not in ORDINAL_COLS and df[col].nunique() > 2

def add_important_interaction(df, important_cols):
    df_new = df.copy()
    vips = [col for col in important_cols if is_numeric(df,col)]
    count = len(vips)
    print(f"combining {count} cols")
    for col1, col2 in combinations(vips, 2):
        df_new[f"{col1}_x_{col2}"] = df[col1] * df[col2]
    print(f"added {(count)*(count-1) // 2} cols as interactions")
    return df_new

def get_xgb_preprocessor():
    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
    ])
    
    # Ordinal Encoder
    ordinal_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('encoder', OrdinalEncoder(categories=ORDINAL_CATEGORIES, handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    # OHE for Nominal
    nominal_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])
    
    # OHE for Bool 
    bool_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')) 
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipe, NUMERIC_COLS),
            ('ord', ordinal_pipe, ORDINAL_COLS),
            ('nom', nominal_pipe, NOMINAL_COLS),
            ('bool', bool_pipe, BOOL_COLS)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform='pandas')

    return preprocessor

def get_lgbm_preprocessor():
    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
    ])
    
    # Ordinal Encoder
    ordinal_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('encoder', OrdinalEncoder(categories=ORDINAL_CATEGORIES, handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    # OHE for Nominal 
    nominal_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)) 
    ])
    
    # OHE for Bool
    bool_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)) 
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipe, NUMERIC_COLS),
            ('ord', ordinal_pipe, ORDINAL_COLS),
            ('nom', nominal_pipe, NOMINAL_COLS), 
            ('bool', bool_pipe, BOOL_COLS)    
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform='pandas')

    return preprocessor

def get_cat_preprocessor():
    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
    ])
    
    # Ordinal encoder
    ordinal_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('encoder', OrdinalEncoder(categories=ORDINAL_CATEGORIES, handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    # keep string
    nominal_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')), 
        ('encoder', FunctionTransformer(lambda x: x.astype(str))) 
    ])
    
    # OHE for Bool
    bool_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)) 
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipe, NUMERIC_COLS),
            ('ord', ordinal_pipe, ORDINAL_COLS),
            ('nom', nominal_pipe, NOMINAL_COLS),
            ('bool', bool_pipe, BOOL_COLS)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform='pandas')

    return preprocessor