import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
df = pd.read_csv("titanic/train.csv")
print("\n===== FIRST 5 ROWS =====")
print(df.head())
print("\n===== DATASET INFO =====")
print(df.info())

print("\n===== SHAPE =====")
print(df.shape)

print("\n===== COLUMN NAMES =====")
print(df.columns)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())


df['Age'] = df['Age'].fillna(df['Age'].median())


df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df.drop('Cabin', axis=1, inplace=True)
print("\n===== AFTER CLEANING =====")
print(df.isnull().sum())
print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.show()
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Passenger Class")
plt.show()
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()
plt.figure(figsize=(8,5))
sns.histplot(df['Fare'], bins=30, kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.show()
plt.figure(figsize=(8,4))
sns.boxplot(x=df['Fare'])
plt.title("Fare Outliers")
plt.show()

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(10,6))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)
plt.title("Correlation Heatmap")
plt.show()
plt.figure(figsize=(8,5))
sns.boxplot(x='Survived', y='Age', data=df)
plt.title("Age vs Survival")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='Survived', y='Fare', data=df)
plt.title("Fare vs Survival")
plt.show()

survival_rate = df.groupby('Pclass')['Survived'].mean()

plt.figure(figsize=(6,4))
survival_rate.plot(kind='bar')
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Average Survival Rate")
plt.show()

df.to_csv("cleaned_titanic.csv", index=False)

print("\nCleaned dataset saved as 'cleaned_titanic.csv'")


print("\n===== KEY INSIGHTS =====")
print("1. Missing values handled successfully.")
print("2. Cabin column removed due to many missing values.")
print("3. Female passengers had higher survival rates.")
print("4. First-class passengers survived more frequently.")
print("5. Fare contains some outliers.")
print("6. Most passengers were between 20 and 40 years old.")
print("7. Passenger class and fare influenced survival.")
print("8. Cleaned dataset exported successfully.")