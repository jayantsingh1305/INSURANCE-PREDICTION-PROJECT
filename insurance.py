import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import warnings 

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

numeric_columns = ['age','bmi','children','charges']
for col in numeric_columns:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col],kde=True,bins=20)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    print(f"Plotting histogram for: {col}")
    plt.show()

# Countplot for categorical column
plt.figure(figsize=(6,4))
sns.countplot(x=df['children'])
plt.title('Count of Children')
plt.xlabel('Number of Children')
plt.ylabel('Count')
plt.show()

# Countplot for sex
plt.figure(figsize=(6,4))
sns.countplot(x=df['sex'])
plt.title('Count of Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

# Countplot for smoker
plt.figure(figsize=(6,4))
sns.countplot(x=df['smoker'])
plt.title('Count of Smoker')
plt.xlabel('Smoker')
plt.ylabel('Count')
plt.show()

for col in numeric_columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.title('Correlation Heatmap')
plt.show()