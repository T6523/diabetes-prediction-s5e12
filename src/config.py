import os

NUMERIC_COLS = ['age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week', 'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides']
NOMINAL_COLS = ['gender', 'ethnicity', 'employment_status']
ORDINAL_COLS = ['education_level', 'income_level', 'smoking_status']
BOOL_COLS = ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history', 'is_external']
TARGET = 'diagnosed_diabetes'

ORDINAL_CATEGORIES = [['No formal','Highschool','Graduate','Postgraduate'], \
                    ['Low','Lower-Middle','Middle','Upper-Middle','High'], \
                    ['Never','Current','Former']]

TO_INTERACT_COLS = ['family_history_diabetes_1', 'Relative_Activity', 'Age_BMI_Risk', 'Chronic_Metabolic_Load', 'physical_activity_minutes_per_week', 'Diet_Activity_Score', 'age', 'Age_WHR_Risk', 'Lipid_Accumulation', 'triglycerides']
OPTIMAL_COLS = ['age', 'physical_activity_minutes_per_week', 'heart_rate', 'triglycerides', 'Cholesterol_Ratio', 'Diet_Activity_Score', 'Relative_Activity', 'Age_BMI_Risk', 'Age_BP_Risk', 'Age_WHR_Risk', 'family_history_diabetes_1', 'cardiovascular_history_1', 'Relative_Activity_x_Diet_Activity_Score', 'Age_BMI_Risk_x_Chronic_Metabolic_Load', 'Age_BMI_Risk_x_Lipid_Accumulation', 'Chronic_Metabolic_Load_x_age', 'physical_activity_minutes_per_week_x_Diet_Activity_Score', 'age_x_Age_WHR_Risk', 'age_x_Lipid_Accumulation', 'Age_WHR_Risk_x_Lipid_Accumulation']

RATIO = 0.6049

XGB_PARAMS = {'n_estimators': 546, 'max_depth': 4, 'learning_rate': 0.09522141307403631, 'min_child_weight': 5, 'gamma': 0.2829267954406464, 'subsample': 0.8714391222089948, 'scale_pos_weight': RATIO,  'n_jobs': -1,'random_state': 42,'eval_metric': 'auc','verbosity': 0}
LGBM_PARAMS = {'n_estimators': 241, 'learning_rate': 0.05186210130983924, 'max_depth': 10, 'num_leaves': 925, 'min_data_in_leaf': 650, 'scale_pos_weight': RATIO,'n_jobs': -1,'random_state': 42,'verbose': -1}
CAT_PARAMS = {'iterations': 762, 'depth': 6, 'learning_rate': 0.07494446029772849, 'l2_leaf_reg': 10, 'subsample': 0.8406007197850853,'scale_pos_weight': RATIO,'verbose': 0,'thread_count': os.cpu_count(),'random_state': 42,'allow_writing_files': False}
