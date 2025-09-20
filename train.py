import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from joblib import dump
import os

# Set options to avoid scientific notation in pandas and numpy
pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))
np.set_printoptions(suppress=True)

# Load datasets
df_customers = pd.read_csv("datasets/customers.csv")
df_loans = pd.read_csv("datasets/loans.csv")
df_bureau = pd.read_csv("datasets/bureau_data.csv")

# Shapes of the dataframes
print(df_customers.shape, df_loans.shape, df_bureau.shape)

# Peek at the data
print(df_customers.head())
print(df_loans.head())

# Merge customer and loans data on cust_id
df = pd.merge(df_customers, df_loans, on='cust_id')
print(df.head())

# Merge with bureau data
df = pd.merge(df, df_bureau, on="cust_id")
print(df.head())

# Data info
print(df.info())

# Convert 'default' column to int type
df['default'] = df['default'].astype(int)
print(df.default.value_counts())
print(df.head())

# Train-test split to avoid data leakage before EDA
X = df.drop('default', axis='columns')
y = df['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Recombine into train and test datasets
df_train = pd.concat([X_train, y_train], axis='columns')
df_test = pd.concat([X_test, y_test], axis='columns')

print(df_train.shape, df_test.shape)

# Data cleaning - missing values
print(df_train.isna().sum())
print(df_train.residence_type.unique())

# Fill missing values in 'residence_type' with mode
mode_residence = df_train.residence_type.mode()
print(mode_residence)
df_train.residence_type.fillna(mode_residence, inplace=True)
print(df_train.isna().sum())
df_test.residence_type.fillna(mode_residence, inplace=True)

# Check duplicates
print(df_train.duplicated().sum())

# Continuous and categorical variable separation
columns_continuous = [
    'age', 'income', 'number_of_dependants', 'years_at_current_address',
    'sanction_amount', 'loan_amount', 'processing_fee', 'gst', 'net_disbursement',
    'loan_tenure_months', 'principal_outstanding', 'bank_balance_at_application',
    'number_of_open_accounts', 'number_of_closed_accounts', 'total_loan_months',
    'delinquent_months', 'total_dpd', 'enquiry_count', 'credit_utilization_ratio'
]

columns_categorical = [
    'cust_id', 'gender', 'marital_status', 'employment_status', 'residence_type',
    'city', 'state', 'zipcode', 'loan_id', 'loan_purpose', 'loan_type',
    'disbursal_date', 'installment_start_dt', 'default'
]

# Boxplot code (commented out for now)
# num_plots=len(columns_continuous)
# num_cols=4
# num_rows=(num_plots+num_cols-1)//num_cols
# fig,axes=plt.subplots(num_rows,num_cols,figsize=(5*num_cols,5*num_rows))
# axes=axes.flatten()
# for i,col in enumerate(columns_continuous):
#     sns.boxplot(x=df_train[col],ax=axes[i])
#     axes[i].set_title(col)
# for j in range(i+1,num_rows*num_cols):
#     axes[j].axis('off')
# plt.tight_layout()
# plt.show()

# Histogram code (commented out for now)
# num_plots=len(columns_continuous)
# num_cols=4
# num_rows=(num_plots+num_cols-1)//num_cols
# fig,axes=plt.subplots(num_rows,num_cols,figsize=(5*num_cols,5*num_rows))
# axes=axes.flatten()
# for i,col in enumerate(columns_continuous):
#     sns.histplot(x=df_train[col],ax=axes[i])
#     axes[i].set_title(col)
# for j in range(i+1,num_rows*num_cols):
#     axes[j].axis('off')
# plt.tight_layout()
# plt.show()

# Summary statistics of 'processing_fee'
print(df_train.processing_fee.describe())

# Check max processing fee
print(df_train[df_train.processing_fee == df_train.processing_fee.max()][['loan_amount','processing_fee']])

# Check cases where processing_fee > loan_amount
print(df_train[df_train.processing_fee > df_train.loan_amount][['loan_amount','processing_fee']])

# Check cases where processing_fee > 3% of loan_amount
print(df_train[(df_train.processing_fee / df_train.loan_amount) > 0.03][['loan_amount','processing_fee']])

# Remove rows where processing_fee is more than 3% of loan_amount
df_train_1 = df_train[(df_train.processing_fee / df_train.loan_amount) < 0.03].copy()

# Double check for any remaining violations
print(df_train_1[(df_train_1.processing_fee / df_train_1.loan_amount) > 0.03][['loan_amount','processing_fee']])

# Apply same cleaning to test data
df_test = df_test[df_test.processing_fee / df_test.loan_amount < 0.03].copy()
print(df_test.shape)

# Look at unique values in categorical columns
for col in columns_categorical:
    print(col, "-", df_train[col].unique())

# Fix spelling mistake in loan_purpose column
df_train_1['loan_purpose'] = df_train_1['loan_purpose'].replace('Personaal','Perosnal')
df_test['loan_purpose'] = df_test['loan_purpose'].replace('Personaal','Perosnal')

print(df_train_1.loan_purpose.unique())


# EDA
# Check customers where GST is more than 20% of loan amount
print(df_train[(df_train.gst/df_train.loan_amount) > 0.2])

# KDE plots for continuous features (commented out version kept for reference)
# plt.figure(figsize=(24,20))  
# for i, col in enumerate(columns_continuous):
#     plt.subplot(6,4,i+1)  # 6 rows, 4 columns, ith subplot
#     sns.kdeplot(df_train_1[df_train_1.default==0][col], fill=True, label='Default=0')
#     sns.kdeplot(df_train_1[df_train_1.default==1][col], fill=True, label='Default=1')
#     plt.title(col)
#     plt.xlabel('')
# plt.tight_layout()
# plt.show()

# ----------------- INSIGHTS -----------------
# Columns like loan_tenure_months, delinquent_months, total_dpd, and credit_utilization_ratio 
# show that higher values indicate a higher likelihood of default, making them strong predictors.
# The remaining columns do not provide obvious insights into default behavior.
# loan_amount and income individually did not appear as strong predictors, 
# but when combined into a ratio (loan_amount / income), they may have stronger influence.

# ----------------- FEATURE ENGINEERING (TRAIN) -----------------

# Print loan_amount and income
print(df_train_1[['loan_amount','income']])

# Create loan_to_income ratio
df_train_1['loan_to_income'] = round(df_train_1['loan_amount']/df_train_1['income'], 2)
print(df_train_1['loan_to_income'].describe())

# KDE plots for loan_to_income (commented out)
# sns.kdeplot(df_train_1[df_train_1.default==0]['loan_to_income'], fill=True, label='Default=0')
# sns.kdeplot(df_train_1[df_train_1.default==1]['loan_to_income'], fill=True, label='Default=1')
# plt.title('loan_to_income')
# plt.xlabel('')
# plt.show()

# Print delinquent months and loan months
print(df_train_1[['delinquent_months','total_loan_months']])

# Create delinquency_ratio = % of months delinquent
df_train_1['delinquency_ratio'] = (df_train_1['delinquent_months'] * 100 / df_train_1['total_loan_months']).round(1)
print(df_train_1[['delinquent_months','total_loan_months','delinquency_ratio']])

# KDE plots for delinquency_ratio (commented out)
# sns.kdeplot(df_train_1[df_train_1.default==0]['delinquency_ratio'], fill=True, label='Default=0')
# sns.kdeplot(df_train_1[df_train_1.default==1]['delinquency_ratio'], fill=True, label='Default=1')
# plt.title('delinquency_ratio')
# plt.xlabel('')
# plt.show()

# ----------------- FEATURE ENGINEERING (TEST) -----------------

# Print loan_amount and income
print(df_test[['loan_amount','income']])

# Create loan_to_income ratio
df_test['loan_to_income'] = round(df_test['loan_amount'] / df_test['income'], 2)
print(df_test['loan_to_income'].describe())

# KDE plots for loan_to_income (commented out)
# sns.kdeplot(df_test[df_test.default==0]['loan_to_income'], fill=True, label='Default=0')
# sns.kdeplot(df_test[df_test.default==1]['loan_to_income'], fill=True, label='Default=1')
# plt.title('loan_to_income')
# plt.xlabel('')
# plt.show()

# Print delinquent months and loan months
print(df_test[['delinquent_months','total_loan_months']])

# Create delinquency_ratio = % of months delinquent
df_test['delinquency_ratio'] = (df_test['delinquent_months'] * 100 / df_test['total_loan_months']).round(1)
print(df_test[['delinquent_months','total_loan_months','delinquency_ratio']])

# KDE plots for delinquency_ratio (commented out)
# sns.kdeplot(df_test[df_test.default==0]['delinquency_ratio'], fill=True, label='Default=0')
# sns.kdeplot(df_test[df_test.default==1]['delinquency_ratio'], fill=True, label='Default=1')
# plt.title('delinquency_ratio')
# plt.xlabel('')
# plt.show()

# ----------------- FEATURE ENGINEERING: AVERAGE DPD -----------------

# For train → average DPD per delinquency month
df_train_1['avg_dpd_per_delinquency'] = np.where(
    df_train_1['delinquent_months'] != 0,
    (df_train_1['total_dpd'] / df_train_1['delinquent_months']).round(1),
    0
)
print(df_train_1['avg_dpd_per_delinquency'].describe())
print(df_train_1['avg_dpd_per_delinquency'].isna().sum())

# For test → average DPD per delinquency month
df_test['avg_dpd_per_delinquency'] = np.where(
    df_test['delinquent_months'] != 0,
    (df_test['total_dpd'] / df_test['delinquent_months']).round(1),
    0
)
print(df_test['avg_dpd_per_delinquency'].describe())
print(df_test['avg_dpd_per_delinquency'].isna().sum())

# KDE plots for avg_dpd_per_delinquency (commented out)
# sns.kdeplot(df_train_1[df_train_1.default==0]['avg_dpd_per_delinquency'], fill=True, label='Default=0')
# sns.kdeplot(df_train_1[df_train_1.default==1]['avg_dpd_per_delinquency'], fill=True, label='Default=1')
# plt.title('avg_dpd_per_delinquency')
# plt.xlabel('')
# plt.show()

# sns.kdeplot(df_test[df_train_1.default==0]['avg_dpd_per_delinquency'], fill=True, label='Default=0')
# sns.kdeplot(df_test[df_train_1.default==1]['avg_dpd_per_delinquency'], fill=True, label='Default=1')
# plt.title('avg_dpd_per_delinquency')
# plt.xlabel('')
# plt.show()

# ----------------- FEATURE SELECTION -----------------

# Check columns
print(df_train_1.columns)

# Drop ID columns
df_train_2 = df_train_1.drop(['cust_id','loan_id'], axis='columns')
df_test = df_test.drop(['cust_id','loan_id'], axis='columns')

# Drop unnecessary raw columns after feature engineering
df_train_3 = df_train_2.drop(
    ['disbursal_date', 'installment_start_dt', 'loan_amount', 'income',
     'total_loan_months','total_dpd', 'delinquent_months'], 
    axis=1
)

df_test = df_test.drop(
    ['disbursal_date', 'installment_start_dt', 'loan_amount', 'income',
     'total_loan_months','total_dpd', 'delinquent_months'], 
    axis=1
)

print(df_train_3.columns)

# ----------------- FEATURE SCALING -----------------

# Select numerical columns
num_cols = df_train_3.select_dtypes(['int64','float64']).columns
print("Numeric columns:", num_cols)

# Split train into features & target
X_train = df_train_3.drop('default', axis='columns')
y_train = df_train_3['default']

# Remove target column from scaling list
cols_to_scale = [col for col in num_cols if col != 'default']
print("Columns to scale:", cols_to_scale)

# Scale numerical columns using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])

# Preview scaled train data
print(X_train.head(3))
print(X_train.describe())

# Apply same scaling to test
X_test = df_test.drop('default', axis='columns')
y_test = df_test['default']
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

# Preview scaled test data
print(X_test.head(3))
print(X_test.describe())

from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt

def calculate_vif(data):
    vif_df = pd.DataFrame()
    vif_df['Column'] = data.columns
    vif_df['VIF'] = [variance_inflation_factor(data.values, i) 
                     for i in range(data.shape[1])]
    return vif_df

# Run VIF calculation on scaled numeric columns
print(calculate_vif(X_train[cols_to_scale]))

# Drop features with high VIF
features_to_drop_vif = ['sanction_amount','processing_fee','gst',
                        'net_disbursement','principal_outstanding']
X_train_1 = X_train.drop(features_to_drop_vif, axis='columns')

# Recompute VIF
numeric_cols = X_train_1.select_dtypes(['int64','float64']).columns
vif_df = calculate_vif(X_train_1[numeric_cols])
print(vif_df)

# Selected numeric features after VIF filtering
selected_numeric_features_vif = vif_df['Column'].values
print(selected_numeric_features_vif)

# ---- Correlation heatmap ----
plt.figure(figsize=(12,12))

# Add 'default' to numeric_cols safely
corr_cols = numeric_cols.union(['default'])

cm = df_train_3[corr_cols].corr()

# import seaborn as sns
# sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm", square=True)

# plt.xticks(rotation=45, ha='right')
# plt.yticks(rotation=0)
# plt.title("Correlation Heatmap (After VIF Feature Selection)")
# plt.tight_layout()
# plt.show()

# woe and IV
# Combine features and target for analysis
temp = pd.concat([X_train_1, y_train], axis=1)
print(temp.groupby('loan_purpose')['default'].agg(['count','sum']))

# ----------------- WOE & IV FUNCTION -----------------
def calculate_woe_iv(df, feature, target):
    # Group by feature categories and compute counts & defaults
    grouped = df.groupby(feature)[target].agg(['count','sum'])
    
    # Treat 'sum' as good (non-defaults) → ⚠️ check mapping of your target!
    grouped = grouped.rename(columns={'count':'total', 'sum':'good'})
    grouped['bad'] = grouped['total'] - grouped['good']
    
    # Calculate totals
    total_good = grouped['good'].sum()
    total_bad = grouped['bad'].sum()
    
    # Calculate percentages
    grouped['good_pct'] = grouped['good'] / total_good
    grouped['bad_pct'] = grouped['bad'] / total_bad
    
    # Add epsilon to avoid log(0)
    epsilon = 1e-6
    grouped['woe'] = np.log((grouped['good_pct'] + epsilon) / (grouped['bad_pct'] + epsilon))
    
    # Information Value = (distribution difference) * WOE
    grouped['iv'] = (grouped['good_pct'] - grouped['bad_pct']) * grouped['woe']
    
    # Replace infinities with 0
    grouped['woe'] = grouped['woe'].replace([np.inf, -np.inf], 0)
    grouped['iv'] = grouped['iv'].replace([np.inf, -np.inf], 0)
    
    # Total IV for the feature
    total_iv = grouped['iv'].sum()
    
    return grouped, total_iv

# ----------------- IV FOR ONE FEATURE -----------------
grouped, total_iv = calculate_woe_iv(
    pd.concat([X_train_1, y_train], axis=1),
    'loan_purpose',
    'default'
)
print(grouped)   # category-level WOE/IV
print(total_iv)  # total IV for loan_purpose

# ----------------- IV FOR ALL FEATURES -----------------
iv_values = {}
for feature in X_train_1.columns:
    if X_train_1[feature].dtype == 'object':  
        # Categorical → directly calculate WOE/IV
        _, iv = calculate_woe_iv(pd.concat([X_train_1, y_train], axis=1), feature, 'default')
    else:  
        # Continuous → first bin into 10 intervals
        X_binned = pd.cut(X_train_1[feature], bins=10, labels=False)
        _, iv = calculate_woe_iv(pd.concat([X_binned, y_train], axis=1), feature, 'default')
    
    iv_values[feature] = iv

print(iv_values)

# ----------------- CREATE IV DATAFRAME -----------------
iv_df = pd.DataFrame(list(iv_values.items()), columns=['Feature', 'IV'])
iv_df = iv_df.sort_values(by='IV', ascending=False)
print(iv_df)   # Features ranked by predictive power

# ----------------- SELECT STRONG FEATURES -----------------
# Rule of thumb: IV > 0.02 = useful predictor
selected_features_iv = [feature for feature, iv in iv_values.items() if iv > 0.02]
print(selected_features_iv)

# ----------------- FEATURE ENCODING -----------------
# Keep only selected features
X_train_reduced = X_train_1[selected_features_iv]
X_test_reduced = X_test[selected_features_iv]

# One-hot encoding for categorical features
X_train_encoded = pd.get_dummies(X_train_reduced, drop_first=True)
X_test_encoded = pd.get_dummies(X_test_reduced, drop_first=True)

X_train_encoded.head()


# attempt 1
# try various models without any sampling (no handelling of class imbalacne)


# model training
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
model=LogisticRegression()
model.fit(X_train_encoded,y_train)
y_pred=model.predict(X_test_encoded)
report=classification_report(y_test,y_pred)
print(report)


from sklearn.ensemble import RandomForestClassifier
# Train RandomForest
model = RandomForestClassifier()
model.fit(X_train_encoded, y_train)
# Predictions
y_pred = model.predict(X_test_encoded)
# Report
report = classification_report(y_test, y_pred)
print(report)


from xgboost import XGBClassifier
# Train XGBoost
model = XGBClassifier()
model.fit(X_train_encoded, y_train)
# Predictions
y_pred = model.predict(X_test_encoded)
# Report
report = classification_report(y_test, y_pred)
print(report)

# from sklearn.model_selection import RandomizedSearchCV
# # Parameter distribution
# param_dist = {
#     'C': np.logspace(-4, 4, 20),   # Logarithmically spaced values
#     'solver': ['lbfgs', 'saga','liblinear', 'newton-cg'],  # Solvers
# }
# # Logistic Regression model
# log_reg = LogisticRegression(max_iter=10000)
# # Randomized Search
# random_search = RandomizedSearchCV(
#     estimator=log_reg,
#     param_distributions=param_dist,
#     n_iter=50,          # number of parameter combinations to try
#     scoring='f1',       # use f1-score
#     cv=3,               # 3-fold cross-validation
#     verbose=2,
#     random_state=42,
#     n_jobs=-1           # use all CPU cores
# )
# # Fit
# random_search.fit(X_train_encoded, y_train)
# # Best parameters & score
# print("Best Parameters:", random_search.best_params_)
# print("Best CV Score:", random_search.best_score_)
# # Best model
# best_model = random_search.best_estimator_
# # Predictions
# y_pred = best_model.predict(X_test_encoded)
# # Report
# print(classification_report(y_test, y_pred))

# from scipy.stats import uniform, randint
# from sklearn.model_selection import RandomizedSearchCV

# # Define parameter distribution for RandomizedSearchCV
# param_dist = {
#     'n_estimators': [100, 150, 200, 250, 300],
#     'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
#     'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
#     'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
#     'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
#     'scale_pos_weight': [1, 2, 3, 5, 7, 10],
#     'reg_alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],  # L1 regularization term
#     'reg_lambda': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]  # L2 regularization term
# }

# xgb = XGBClassifier()

# random_search = RandomizedSearchCV(estimator=xgb, param_distributions=param_dist, n_iter=100,
#                                    scoring='f1', cv=3, verbose=1, n_jobs=-1, random_state=42)

# random_search.fit(X_train_encoded, y_train)

# # Print the best parameters and best score
# print(f"Best Parameters: {random_search.best_params_}")
# print(f"Best Score: {random_search.best_score_}")

# best_model = random_search.best_estimator_
# y_pred = best_model.predict(X_test_encoded)
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# attempt 2 
# trying various models with undersampling 
print(X_train_1.value_counts(),y_train.value_counts())
from imblearn.under_sampling import RandomUnderSampler
rus=RandomUnderSampler(random_state=42)
X_train_rus,y_train_rus=rus.fit_resample(X_train_encoded,y_train)
print(X_train_rus.value_counts(),y_train_rus.value_counts())

model=LogisticRegression()
model.fit(X_train_rus,y_train_rus)
y_pred=model.predict(X_test_encoded)
report=classification_report(y_test,y_pred)
print(report)

model = XGBClassifier()
model.fit(X_train_rus,y_train_rus)
# Predictions
y_pred = model.predict(X_test_encoded)
# Report
report = classification_report(y_test, y_pred)
print(report)

# attempt 3
# logistic regression ,with oversampling using SMOTE Tomek,parameter tuning using optuna
from imblearn.combine import SMOTETomek

smt=SMOTETomek()
X_train_smt,y_train_smt=smt.fit_resample(X_train_encoded,y_train)

print(X_train_smt.value_counts(),y_train_smt.value_counts())

model=LogisticRegression()
model.fit(X_train_smt,y_train_smt)
y_pred=model.predict(X_test_encoded)
report=classification_report(y_test,y_pred)
print(report)

import optuna
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import cross_val_score

def objective(trial):
    param = {
        'C': trial.suggest_float('C', 1e-4, 1e4, log=True),
        'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga', 'newton-cg']),
        'tol': trial.suggest_float('tol', 1e-6, 1e-1, log=True),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
    }
    try:
        model = LogisticRegression(**param, max_iter=10000)
        f1_scorer = make_scorer(f1_score, average='macro')
        scores = cross_val_score(model, X_train_smt, y_train_smt, cv=3,
                                 scoring=f1_scorer, n_jobs=-1)
        return np.mean(scores)
    except Exception:
        return -1.0  # return poor score if invalid combo

study_logistic = optuna.create_study(direction='maximize')
study_logistic.optimize(objective, n_trials=50)
# Retrieve best parameters
best_params = study_logistic.best_params

# Create and fit the best model on full training data
best_model_logistic = LogisticRegression(
    **best_params,
    max_iter=10000
)
best_model_logistic.fit(X_train_smt, y_train_smt)

# Predict on test set
y_pred = best_model_logistic.predict(X_test_encoded)

# Evaluate
print(classification_report(y_test, y_pred))


# # attempt 4 
# # xgboost ,handle class imbalance usinf SMOTETomek tune using optuna

# # Define the objective function for Optuna
# def objective(trial):
#     param = {
#         'objective': 'binary:logistic',
#         'eval_metric': 'logloss',
#         'verbosity': 0,
#         'booster': 'gbtree',
#         'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
#         'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
#         'subsample': trial.suggest_float('subsample', 0.4, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'eta': trial.suggest_float('eta', 0.01, 0.3),
#         'gamma': trial.suggest_float('gamma', 0, 10),
#         'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#         'max_delta_step': trial.suggest_int('max_delta_step', 0, 10)
#     }

#     model = XGBClassifier(**param)
    
#     # Calculate the cross-validated f1_score
#     f1_scorer = make_scorer(f1_score, average='macro')
#     scores = cross_val_score(model, X_train_smt, y_train_smt, cv=3, scoring=f1_scorer, n_jobs=-1)
    
#     return np.mean(scores)

# study_xgb = optuna.create_study(direction='maximize')
# study_xgb.optimize(objective, n_trials=50)

# print('Best trial:')
# trial = study_xgb.best_trial
# print('  F1-score: {}'.format(trial.value))
# print('  Params: ')
# for key, value in trial.params.items():
#     print('    {}: {}'.format(key, value))
    
# best_params = study_xgb.best_params
# best_model_xgb = XGBClassifier(**best_params)
# best_model_xgb.fit(X_train_smt, y_train_smt)

# # Evaluate on the test set
# y_pred = best_model_xgb.predict(X_test_encoded)

# report = classification_report(y_test, y_pred)
# print(report)


y_pred = best_model_logistic.predict(X_test_encoded)

# Evaluate
print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_curve,auc
probabilities=best_model_logistic.predict_proba(X_test_encoded)[:,1]
print(probabilities)

fpr,tpr,thresholds=roc_curve(y_test,probabilities)
print(fpr[:10],tpr[:10],thresholds[:10])
area=auc(fpr,tpr)
print(area)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % area)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # diagonal
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# -------------------------------
# PREDICTIONS ON TEST DATA
# -------------------------------
# probabilities for the positive class (Default=1)
probabilities = best_model_logistic.predict_proba(X_test_encoded)[:, 1]

# create evaluation DataFrame
df_eval = pd.DataFrame({
    'Default Truth': y_test,             # actual labels
    'Default Probability': probabilities # predicted probabilities
})

# -------------------------------
# DECILE ASSIGNMENT
# -------------------------------
# divide data into 10 deciles based on predicted probability
df_eval.loc[:, 'Decile'] = pd.qcut(df_eval['Default Probability'], 10, labels=False, duplicates='drop')

# check statistics of a specific decile (optional)
print(df_eval[df_eval.Decile==8]['Default Probability'].describe())

# -------------------------------
# DECILE AGGREGATION
# -------------------------------
df_decile = df_eval.groupby('Decile').apply(lambda x: pd.Series({
    'Minimum Probability': x['Default Probability'].min(),
    'Maximum Probability': x['Default Probability'].max(),
    'Events': x['Default Truth'].sum(),
    'Non-events': x['Default Truth'].count() - x['Default Truth'].sum()
})).reset_index()

# calculate event and non-event rates per decile
df_decile['Event Rate'] = df_decile['Events'] * 100 / (df_decile['Events'] + df_decile['Non-events'])
df_decile['Non-event Rate'] = df_decile['Non-events'] * 100 / (df_decile['Events'] + df_decile['Non-events'])

# -------------------------------
# CUMULATIVE CALCULATIONS FOR KS
# -------------------------------
# sort deciles from highest to lowest probability
df_decile = df_decile.sort_values(by='Decile', ascending=False).reset_index(drop=True)

# cumulative events and non-events
df_decile['Cum Events'] = df_decile['Events'].cumsum()
df_decile['Cum Non-events'] = df_decile['Non-events'].cumsum()

# cumulative event and non-event rates
df_decile['Cum Event Rate'] = df_decile['Cum Events'] * 100 / df_decile['Events'].sum()
df_decile['Cum Non-event Rate'] = df_decile['Cum Non-events'] * 100 / df_decile['Non-events'].sum()

# KS statistic: max difference between cumulative rates
df_decile['KS'] = abs(df_decile['Cum Event Rate'] - df_decile['Cum Non-event Rate'])

print("KS per decile:")
print(df_decile[['Decile', 'KS']])
print("Max KS:", df_decile['KS'].max())

# -------------------------------
# GINI COEFFICIENT
# -------------------------------
# If you already have AUC (area under ROC curve)
# gini = 2 * AUC - 1
gini_coefficient = 2 * area - 1
print("AUC:", area)
print("Gini Coefficient:", gini_coefficient)

# -------------------------------
# FEATURE IMPORTANCE (LOGISTIC REGRESSION)
# -------------------------------
final_model = best_model_logistic
feature_importance = final_model.coef_[0]

# create DataFrame for visualization
coef_df = pd.DataFrame(feature_importance, index=X_train_encoded.columns, columns=['Coefficients'])
coef_df = coef_df.sort_values(by='Coefficients', ascending=True)

# plot horizontal bar chart
plt.figure(figsize=(8, 4))
plt.barh(coef_df.index, coef_df['Coefficients'], color='steelblue')
plt.xlabel('Coefficient Value')
plt.title('Feature Importance in Logistic Regression')
plt.show()

# -------------------------------
# SAVE MODEL FOR PRODUCTION
# -------------------------------
# ensure artifacts folder exists
os.makedirs('artifacts', exist_ok=True)

model_data = {
    'model': final_model,              # trained logistic regression model
    'features': X_train_encoded.columns, # feature columns
    'scaler': scaler,                  # scaler used for numerical features
    'cols_to_scale': cols_to_scale     # columns that were scaled
}
dump(model_data, 'artifacts/model_data.joblib')

# -------------------------------
# DISPLAY MODEL COEFFICIENTS AND INTERCEPT
# -------------------------------
print("Coefficients:", final_model.coef_)
print("Intercept:", final_model.intercept_)