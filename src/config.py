NUMERIC_COLS = ['age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week', 'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides']
NOMINAL_COLS = ['gender', 'ethnicity', 'employment_status']
ORDINAL_COLS = ['education_level', 'income_level', 'smoking_status']
BOOL_COLS = ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']
TARGET = 'diagnosed_diabetes'

ORDINAL_CATEGORIES = [['No formal','Highschool','Graduate','Postgraduate'], \
                    ['Low','Lower-Middle','Middle','Upper-Middle','High'], \
                    ['Never','Current','Former']]