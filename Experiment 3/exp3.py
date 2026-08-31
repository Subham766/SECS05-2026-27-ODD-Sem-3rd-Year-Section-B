import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# 1. LOAD DATASET
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

print("=" * 70)
print("STATISTICAL ANALYSIS AND HYPOTHESIS TESTING")
print("=" * 70)

# 2. DISPLAY DATA
print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Dimensions:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 3. DESCRIPTIVE STATISTICS
print("\nDescriptive Statistics:")

numeric_columns = [
    "MonthlyIncome",
    "YearsAtCompany",
    "JobSatisfaction",
    "Age"
]

for column in numeric_columns:
    if column in df.columns:
        print("\n", column)
        print("Mean:", df[column].mean())
        print("Median:", df[column].median())
        print("Mode:", df[column].mode()[0])
        print("Variance:", df[column].var())
        print("Standard Deviation:", df[column].std())

# 4. CORRELATION ANALYSIS
print("\nPearson Correlation:")

correlation_columns = [
    "Age",
    "MonthlyIncome",
    "YearsAtCompany",
    "JobSatisfaction"
]

correlation = df[correlation_columns].corr(method="pearson")

print(correlation)

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix")
plt.show()

# 5. HYPOTHESIS TESTING
print("\nIndependent Sample T-Test")

# H0: No significant difference in monthly income
# H1: Significant difference in monthly income

stay = df[df["Attrition"] == "No"]["MonthlyIncome"]
leave = df[df["Attrition"] == "Yes"]["MonthlyIncome"]

t_stat, p_value = stats.ttest_ind(
    stay,
    leave,
    equal_var=False
)

print("T-statistic:", t_stat)
print("P-value:", p_value)

if p_value < 0.05:
    print("Reject H0: Significant difference exists.")
else:
    print("Fail to reject H0: No significant difference found.")

# 6. ONE-WAY ANOVA
print("\nOne-Way ANOVA")

groups = [
    group["JobSatisfaction"].dropna()
    for name, group in df.groupby("JobLevel")
]

f_stat, anova_p = stats.f_oneway(*groups)

print("F-statistic:", f_stat)
print("P-value:", anova_p)

if anova_p < 0.05:
    print("Reject H0: Significant difference exists among groups.")
else:
    print("Fail to reject H0: No significant difference among groups.")

# 7. LINEAR REGRESSION
print("\nLinear Regression")

X = df["YearsAtCompany"]
y = df["MonthlyIncome"]

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print(model.summary())

print("\nRegression Coefficients:")
print(model.params)

print("\nR-squared:", model.rsquared)

# 8. REGRESSION PLOT
plt.figure(figsize=(8, 5))

sns.regplot(
    data=df,
    x="YearsAtCompany",
    y="MonthlyIncome"
)

plt.title("Years at Company vs Monthly Income")
plt.xlabel("Years at Company")
plt.ylabel("Monthly Income")

plt.show()

# 9. FINAL OBSERVATIONS
print("\n" + "=" * 70)
print("KEY OBSERVATIONS")
print("=" * 70)

print("\n1. T-Test P-value:", p_value)

if p_value < 0.05:
    print("Monthly income differs significantly between employees who leave and stay.")
else:
    print("No statistically significant difference in monthly income was found.")

print("\n2. ANOVA P-value:", anova_p)

if anova_p < 0.05:
    print("Job satisfaction differs significantly among job-level groups.")
else:
    print("No statistically significant difference was found among job-level groups.")

print("\n3. Regression R²:", model.rsquared)

print("\nStatistical analysis completed successfully!")


# Questions and Answers
# 1. What is the difference between descriptive statistics and inferential statistics?
# Answer:
# Descriptive statistics summarize and describe the main features of collected data using measures such as mean, median, mode, variance, and standard deviation. Inferential statistics use sample data to make conclusions or predictions about a larger population.

# 2. Explain the concepts of the Null Hypothesis (H₀) and Alternative Hypothesis (H₁).
# Answer:
# The Null Hypothesis (H₀) states that there is no significant difference or relationship between variables. The Alternative Hypothesis (H₁) states that a significant difference or relationship exists. Statistical tests are used to decide whether H₀ should be rejected.

# 3. What is a p-value? How is it used to make statistical decisions?
# Answer:
# A p-value indicates how likely the observed result is if the null hypothesis is true. Generally, if p < 0.05, H₀ is rejected and the result is considered statistically significant. If p ≥ 0.05, H₀ is not rejected.

# 4. Differentiate between a t-test and ANOVA. In which situations is each test applied?
# Answer:
# A t-test is generally used to compare the means of two groups. ANOVA is used to compare the means of three or more groups. For example, a t-test can compare income between employees who leave and stay, while ANOVA can compare satisfaction across multiple groups.

# 5. What does the Pearson correlation coefficient indicate? What are its possible values?
# Answer:
# Pearson correlation measures the strength and direction of the linear relationship between two numerical variables. Its value ranges from -1 to +1. A positive value indicates a positive relationship, while a negative value indicates a negative relationship. A value near zero indicates little or no linear relationship.

# 6. Explain the significance of R² (Coefficient of Determination) in Linear Regression.
# Answer:
# R² represents the proportion of variation in the dependent variable that is explained by the independent variable(s). Its value generally ranges from 0 to 1. A higher R² indicates that the regression model explains more of the variation in the dependent variable.

# 7. Why is statistical analysis important before applying machine learning algorithms?
# Answer:
# Statistical analysis helps understand the data, identify relationships, detect anomalies, check assumptions, and select useful variables. This improves data quality and helps build more reliable machine learning models.

# 8. What assumptions should be satisfied before performing a t-test or ANOVA?
# Answer:
# Important assumptions include independent observations, approximately normally distributed data within groups, and similar variances between groups. For ANOVA, the observations should also be independent and the groups should be appropriately defined.

# 9. How can regression analysis help organizations in forecasting and decision-making?
# Answer:
# Regression analysis identifies relationships between variables and can be used to predict future outcomes. Organizations can use these predictions for planning, forecasting revenue or costs, estimating demand, and making data-driven decisions.

# 10. Give two real-world applications where hypothesis testing is commonly used in data analytics.
# Answer:

# Employee analytics: Testing whether employee income differs significantly between employees who leave and those who stay.
# Business analytics: Testing whether different groups, products, or strategies produce significantly different results.