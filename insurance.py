import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import warnings 
from pathlib import Path
import joblib

warnings.filterwarnings('ignore')

df = pd.read_csv('insurance.csv')

#EDA

print("SHAPE -->",df.shape)
print("------")
print(df.head())
print("------")
print(df.info())
print("------")
print(df.describe())
print("------")
print(df.isnull().sum())
print("------")

# numeric_columns = ['age','bmi','children','charges']
# for col in numeric_columns:
#     plt.figure(figsize=(6,4))
#     sns.histplot(df[col],kde=True,bins=20)
#     plt.title(f'Distribution of {col}')
#     plt.xlabel(col)
#     plt.ylabel('Frequency')
#     print(f"Plotting histogram for: {col}")
#     plt.show()

# plt.figure(figsize=(6,4))
# sns.countplot(x=df['children'])
# plt.title('Count of Children')
# plt.xlabel('Number of Children')
# plt.ylabel('Count')
# plt.show()

# plt.figure(figsize=(6,4))
# sns.countplot(x=df['sex'])
# plt.title('Count of Sex')
# plt.xlabel('Sex')
# plt.ylabel('Count')
# plt.show()

# plt.figure(figsize=(6,4))
# sns.countplot(x=df['smoker'])
# plt.title('Count of Smoker')
# plt.xlabel('Smoker')
# plt.ylabel('Count')
# plt.show()

# for col in numeric_columns:
#     plt.figure(figsize=(6,4))
#     sns.boxplot(x=df[col])
#     plt.title(f'Boxplot of {col}')
#     plt.xlabel(col)
#     plt.show()

# plt.figure(figsize=(8,6))
# sns.heatmap(df.corr(numeric_only=True),annot=True)
# plt.title('Correlation Heatmap')
# plt.show()

#DATA CLEANING AND PREPROCESSING

df_cleaned = df.copy()

print(df_cleaned.head())
print("------")
print(df_cleaned.shape)
print("------")

print(df_cleaned['sex'].value_counts())
df_cleaned['sex']=df_cleaned['sex'].map({"male":0,"female":1})
print(df_cleaned.head())

print(df_cleaned['smoker'].value_counts())
df_cleaned['smoker']=df_cleaned['smoker'].map({"no":0,"yes":1})
print(df_cleaned.head())

df_cleaned.rename(columns={
    'sex':"is_female",
    'smoker':'is_smoker'
},inplace=True)

df_cleaned = pd.get_dummies(df_cleaned,columns=['region'],drop_first=True)
# Convert only boolean dummy columns to int; keep continuous columns as float.
bool_cols = df_cleaned.select_dtypes(include=['bool']).columns
df_cleaned[bool_cols] = df_cleaned[bool_cols].astype(int)
print(df_cleaned)

#FEATURE ENGINEERING AND EXTRACTION
df_cleaned['bmi_category'] = pd.cut(
    df_cleaned['bmi'],
    bins=[0,18.5,24.9,29.9,float('inf')],
    labels=['Underweight','Normal','Overweight','Obese']
)
df_cleaned = pd.get_dummies(df_cleaned,columns=['bmi_category'],drop_first=True)
bool_cols = df_cleaned.select_dtypes(include=['bool']).columns
df_cleaned[bool_cols] = df_cleaned[bool_cols].astype(int)

# Add interaction features useful for downstream models.
df_cleaned['age_bmi_interaction'] = df_cleaned['age'] * df_cleaned['bmi']
df_cleaned['smoker_bmi_interaction'] = df_cleaned['is_smoker'] * df_cleaned['bmi']

print(df_cleaned.head())

print(df_cleaned.columns)

# Feature extraction
target_col = 'charges'
X = df_cleaned.drop(columns=[target_col])
y = df_cleaned[target_col]

print("------")
print("FEATURE MATRIX SHAPE -->", X.shape)
print("TARGET SHAPE -->", y.shape)
print("------")
print("FEATURE LIST -->")
print(X.columns.tolist())

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

cols = ['age', 'bmi', 'children', 'age_bmi_interaction', 'smoker_bmi_interaction']
scaler = StandardScaler()

X[cols] = scaler.fit_transform(X[cols])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

models = {
    'LinearRegression': LinearRegression(),
    'RandomForestRegressor': RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
}

print("------")
print("MODEL TRAINING AND EVALUATION")
print("------")

evaluation_rows = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"{model_name} Results")
    print(f"MAE  --> {mae:.2f}")
    print(f"RMSE --> {rmse:.2f}")
    print(f"R2   --> {r2:.4f}")
    print("------")

    evaluation_rows.append({
        'model': model_name,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'model_obj': model
    })

results_df = pd.DataFrame(evaluation_rows).sort_values(by='r2', ascending=False)
best_row = results_df.iloc[0]

artifacts_dir = Path('artifacts')
artifacts_dir.mkdir(exist_ok=True)

best_model_path = artifacts_dir / 'best_insurance_model.joblib'
scaler_path = artifacts_dir / 'feature_scaler.joblib'

joblib.dump(best_row['model_obj'], best_model_path)
joblib.dump(scaler, scaler_path)

print(f"Best model selected --> {best_row['model']}")
print(f"Saved model to --> {best_model_path}")
print(f"Saved scaler to --> {scaler_path}")
print("------")

print(df_cleaned.head())

from scipy.stats import pearsonr

## PEARSON CORRELATION CALCULATION

selected_features = [
    'age','bmi','children','is_female','is_smoker',
    'region_northwest','region_southeast','region_southwest',
    'bmi_category_Normal', 'bmi_category_Overweight','bmi_category_Obese'
]

correlations = {
    feature: pearsonr(df_cleaned[feature],df_cleaned['charges'])[0]
    for feature in selected_features
}

correlation_df = pd.DataFrame(list(correlations.items()), columns=['Feature','Pearson Correlation'])
correlation_df = correlation_df.sort_values(by='Pearson Correlation', ascending=False)

print("PEARSON CORRELATION WITH CHARGES")
print(correlation_df)
