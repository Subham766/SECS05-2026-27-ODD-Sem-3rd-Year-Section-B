import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Name: Subham Kr. Sinha 
# Roll no. 57


# 1. Load dataset
df = pd.read_csv("titanic.csv")

# 2. Display dataset
print("First 5 Records:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nStatistical Description:")
print(df.describe())

# 3. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 4. Handle missing values
if "Age" in df.columns:
    df["Age"] = df["Age"].fillna(df["Age"].median())

if "Embarked" in df.columns:
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

if "Fare" in df.columns:
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())


# Name: Subham Kr. Sinha 
# Roll no. 57


# 5. Remove duplicate records
print("\nDuplicate Records:", df.duplicated().sum())
df = df.drop_duplicates()

# 6. Standardize categorical values
if "Sex" in df.columns:
    df["Sex"] = df["Sex"].str.lower().str.strip()

if "Embarked" in df.columns:
    df["Embarked"] = df["Embarked"].str.upper().str.strip()

# 7. One-Hot Encoding
categorical_columns = []

for col in ["Sex", "Embarked"]:
    if col in df.columns:
        categorical_columns.append(col)

if categorical_columns:
    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True
    )


# Name: Subham Kr. Sinha 
# Roll no. 57

# 8. Detect outliers using IQR
if "Fare" in df.columns:
    Q1 = df["Fare"].quantile(0.25)
    Q3 = df["Fare"].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df["Fare"] = df["Fare"].clip(lower, upper)

# 9. Feature Engineering
if "SibSp" in df.columns and "Parch" in df.columns:
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

if "Age" in df.columns:
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 60, np.inf],
        labels=["Child", "Teenager", "Adult", "Senior"]
    )

# Convert AgeGroup to numerical form
if "AgeGroup" in df.columns:
    df["AgeGroup"] = df["AgeGroup"].cat.codes

# 10. Normalize numerical attributes
if "Fare" in df.columns:
    scaler = MinMaxScaler()
    df["Fare"] = scaler.fit_transform(df[["Fare"]])

# 11. Check cleaned data
print("\nCleaned Dataset:")
print(df.head())

print("\nRemaining Missing Values:")
print(df.isnull().sum())

# 12. Visualize an attribute
if "Age" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df["Age"])
    plt.title("Age Distribution After Preprocessing")
    plt.show()


# Name: Subham Kr. Sinha 
# Roll no. 57


# 13. Save cleaned dataset
df.to_csv("cleaned_titanic.csv")

print("\nCleaned dataset saved successfully.")




# Q1. Why is data preprocessing considered one of the most important phases in data analytics?

# Data preprocessing is important because real-world data may contain missing values, duplicate records, inconsistent formats, categorical variables, and outliers. Cleaning and transforming the data improves its quality and reliability before analysis, visualization, or machine learning.

# Q2. Explain different methods of handling missing values with suitable examples.

# Missing values can be handled using several methods:

# Mean: Replace missing numerical values with the average value.
# Median: Replace missing numerical values with the middle value. It is useful when data contains outliers.
# Mode: Replace missing categorical values with the most frequently occurring value.
# Row removal: Remove rows containing missing values.
# Column removal: Remove a column if a very large amount of its data is missing.

# For example, missing Age values in the Titanic dataset can be replaced with the median age.

# Q3. Differentiate between Label Encoding and One-Hot Encoding.
# Label Encoding	One-Hot Encoding
# Converts categories into numerical labels.	Creates separate columns for each category.
# Example: Male = 0, Female = 1.	Example: Male and Female become separate binary columns.
# Uses one column.	Usually creates multiple columns.
# Suitable when categories have an order or when numeric labels are appropriate.	Useful for nominal categories without an inherent order.

# The experiment identifies both Label Encoding and One-Hot Encoding as methods for converting categorical variables into numerical form.

# Q4. What are outliers? How can they affect analytical results?

# Outliers are observations that are unusually high or low compared with most other observations.

# They can:

# Distort the mean.
# Increase standard deviation.
# Affect correlations.
# Influence statistical analysis.
# Reduce the performance of some machine learning models.

# Outliers can be detected using box plots and the IQR method.

# Q5. Explain the difference between normalization and standardization.

# Normalization transforms values into a fixed range, commonly between 0 and 1.

# Formula:

# X' = (X - Xmin) / (Xmax - Xmin)

# Standardization transforms data so that it has a mean of 0 and a standard deviation of 1.

# Formula:

# Z = (X - Mean) / Standard Deviation

# Normalization is useful when a fixed range is required, while standardization is useful when algorithms work better with centered and scaled data.

# The experiment requires numerical attributes to be normalized or standardized to bring them to a common scale.

# Q6. Why should duplicate records be removed before analysis?

# Duplicate records should be removed because they represent repeated observations. If they are not removed, they can:

# Increase the frequency of certain observations.
# Produce biased statistics.
# Affect averages and other calculations.
# Influence machine learning models.
# Reduce data consistency.

# Therefore, duplicate records should be removed before analysis.

# Q7. What is feature engineering? Give two practical examples.

# Feature engineering is the process of creating new useful attributes from existing data.

# Example 1: Family Size

# df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Example 2: Age Group

# Age can be divided into categories such as:

# Child
# Teenager
# Adult
# Senior

# These examples are consistent with the experiment's suggested Family Size and Age Group features.



# Name: Subham Kr. Sinha 
# Roll no. 57



# Q8. Which preprocessing techniques would you apply to the IBM HR Employee Attrition dataset and why?

# The following techniques can be applied:

# Missing-value handling — to deal with incomplete records.
# Duplicate removal — to prevent repeated employee records.
# Categorical encoding — to convert attributes such as department or job role into numerical form.
# Outlier detection — to identify unusually high or low numerical values such as income.
# Normalization/standardization — to bring numerical attributes to a common scale.
# Feature engineering — to create useful attributes such as income categories or age groups.

# These techniques correspond to the preprocessing methods required by the experiment.

# Q9. How does poor-quality data affect machine learning model performance?

# Poor-quality data can negatively affect machine learning models. Missing, incorrect, duplicate, inconsistent, and noisy data can cause the model to learn incorrect patterns.

# As a result:

# Prediction accuracy may decrease.
# Model performance may become unreliable.
# Results may become biased.
# The model may fail to generalize to new data.

# Therefore, data should be properly cleaned and preprocessed before machine learning.


# Name: Subham Kr. Sinha 
# Roll no. 57



# Q10. Name any three Python libraries commonly used for data preprocessing.

# Three commonly used Python libraries are:

# Pandas — data loading, cleaning, and manipulation.
# NumPy — numerical calculations and array operations.
# Scikit-learn — preprocessing, scaling, encoding, and machine learning.

# The experiment specifically lists Pandas and NumPy among its required libraries and also specifies Matplotlib and Seaborn.