# %%
%load_ext autoreload
%autoreload 2

# %%
import sys
import os
sys.path.append(os.path.abspath('../src'))
from data_loader import DiabetesLoader

loader = DiabetesLoader('../data/raw/train.csv')
X_train, y_train, X_val, y_val = loader.get_data()
# %%
from preprocessing import get_preprocessor
# %%
pipeline = get_preprocessor()
# %%
print(type(X_train))
# %%
import pandas as pd

X_test = pd.read_csv('../data/raw/test.csv')
X_test.head()
# %%
X_train_processed = pipeline.fit_transform(X_train)
# %%
if 'id' in X_test.columns:
    id = X_test.pop('id')
# %%
X_val_processed = pipeline.transform(X_val)
X_test_processed = pipeline.transform(X_test)    
# %%
ratio = round(sum(y_train == 0)/ sum(y_train == 1), 4)
# %%
from xgboost import XGBClassifier
# %%
xgb = XGBClassifier(scale_pos_weight=ratio, eval_metric="auc")
xgb.fit(X_train_processed, y_train)
# %%
from sklearn.metrics import classification_report, roc_auc_score
y_pred = xgb.predict(X_val_processed)
print(classification_report(y_val, y_pred))
print(roc_auc_score(y_val, y_pred))
# %%
y_submit = xgb.predict_proba(X_test_processed)[:, 1]
# %%
df_submit = pd.DataFrame(y_submit,columns=[TARGET], index=id)
# %%
display(df_submit.head())
print(df_submit.shape)
# %%
df_submit.to_csv('../outputs/xgb_default.csv')