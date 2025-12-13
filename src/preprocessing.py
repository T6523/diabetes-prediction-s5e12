from config import NUMERIC_COLS, NOMINAL_COLS, ORDINAL_COLS, BOOL_COLS, ORDINAL_CATEGORIES
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler , OrdinalEncoder, OneHotEncoder


def get_preprocessor():

    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())
    ])
    
    ordinal_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('encoder', OrdinalEncoder(categories=ORDINAL_CATEGORIES))
    ])
    
    nominal_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    bool_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('encoder', OneHotEncoder(drop='first')) 
    ])

    preprocessor = ColumnTransformer(
        transformers=[
        ('num', numeric_pipe, NUMERIC_COLS),
        ('ord', ordinal_pipe, ORDINAL_COLS),
        ('nom', nominal_pipe, NOMINAL_COLS),
        ('bool', bool_pipe, BOOL_COLS)
        ],
        remainder='passthrough'
    )

    return preprocessor