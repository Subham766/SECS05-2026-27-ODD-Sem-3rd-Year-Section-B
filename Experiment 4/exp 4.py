import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD DATASET
df = pd.read_csv("superstore_sales.csv")

print("=" * 70)
print("ADVANCED DATA VISUALIZATION")
print("=" * 70)

# 2. DISPLAY DATA
print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Dimensions:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 3. BAR CHART - SALES BY REGION
region_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(
    x=region_sales.index,
    y=region_sales.values
)

plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.show()

# 4. BAR CHART - SALES BY CATEGORY
category_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(
    x=category_sales.index,
    y=category_sales.values
)

plt.title("Sales by Product Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=20)
plt.show()

# 5. LINE CHART - MONTHLY SALES
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

monthly_sales = (
    df.groupby(df["Order_Date"].dt.to_period("M"))["Sales"]
    .sum()
    .reset_index()
)

monthly_sales["Order_Date"] = monthly_sales["Order_Date"].dt.to_timestamp()

plt.figure(figsize=(12, 5))
sns.lineplot(
    data=monthly_sales,
    x="Order_Date",
    y="Sales",
    marker="o"
)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.show()

# 6. HISTOGRAM - SALES DISTRIBUTION
plt.figure(figsize=(8, 5))

sns.histplot(
    df["Sales"],
    bins=15,
    kde=True
)

plt.title("Distribution of Sales")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()

# 7. BOX PLOT - SALES BY REGION
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Region",
    y="Sales"
)

plt.title("Sales Distribution by Region")
plt.xlabel("Region")
plt.ylabel("Sales")
plt.show()

# 8. SCATTER PLOT - SALES VS PROFIT
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Sales",
    y="Profit",
    hue="Category"
)

plt.title("Sales vs Profit")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.show()

# 9. CORRELATION HEATMAP
numeric_columns = [
    "Sales",
    "Profit",
    "Quantity",
    "Discount"
]

correlation = df[numeric_columns].corr()

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()

# 10. SALES BY REGION AND CATEGORY
region_category = (
    df.groupby(["Region", "Category"])["Sales"]
    .sum()
    .reset_index()
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=region_category,
    x="Region",
    y="Sales",
    hue="Category"
)

plt.title("Sales by Region and Category")
plt.xlabel("Region")
plt.ylabel("Sales")
plt.show()

# 11. FINAL OBSERVATIONS
print("\n" + "=" * 70)
print("KEY OBSERVATIONS")
print("=" * 70)

print("\n1. Region with highest sales:")
print(region_sales.idxmax())

print("\n2. Category with highest sales:")
print(category_sales.idxmax())

print("\n3. Highest monthly sales:")
print(
    monthly_sales.loc[
        monthly_sales["Sales"].idxmax(),
        "Order_Date"
    ].strftime("%B %Y")
)

print("\n4. Correlation between Sales and Profit:")
print(correlation.loc["Sales", "Profit"])

print("\nVisualization completed successfully!")



# Questions and Answers
# 1. Why is data visualization considered an essential component of data analytics?
# Answer:
# Data visualization converts raw data into graphical representations. It makes complex data easier to understand and helps identify trends, patterns, outliers, distributions, and relationships.

# 2. Differentiate between Matplotlib and Seaborn. Which library is more suitable for statistical visualizations and why?
# Answer:
# Matplotlib is a general-purpose visualization library that provides detailed control over graphs. Seaborn is built on Matplotlib and provides easier-to-use statistical visualizations with attractive default styles. Seaborn is more suitable for statistical visualizations because it provides functions specifically designed for statistical analysis.

# 3. Which type of chart would you choose to compare sales across different regions? Justify your answer.
# Answer:
# A bar chart is suitable for comparing sales across different regions because it clearly displays the sales value for each categorical region, making comparison easy.

# 4. What information can be obtained from a histogram and a box plot?
# Answer:
# A histogram shows the distribution and frequency of numerical data. A box plot shows the median, spread, quartiles, and possible outliers, and can be used to compare distributions between categories.

# 5. Explain the purpose of a scatter plot. How does it help in identifying relationships between variables?
# Answer:
# A scatter plot displays the relationship between two numerical variables. It helps identify patterns, trends, positive or negative relationships, and possible outliers.

# 6. What is a correlation heatmap? How can it assist in feature selection for machine learning?
# Answer:
# A correlation heatmap is a graphical representation of correlations between numerical variables. It helps identify strongly related features and can assist in selecting useful features while avoiding highly redundant variables.

# 7. Why is chart customization (titles, labels, legends, colors) important in data visualization?
# Answer:
# Chart customization improves readability and makes the information easier to understand. Titles, labels, legends, and appropriate colors help users correctly interpret the visualization.

# 8. What factors should be considered while selecting an appropriate visualization for a dataset?
# Answer:
# The type of data, number of variables, analytical objective, and relationship that needs to be shown should be considered. The selected chart should clearly communicate the required information and make patterns easy to identify.

# 9. How can misleading visualizations affect business decision-making? Give a real-world example.
# Answer:
# Misleading visualizations can make data appear different from reality and may cause incorrect business decisions. For example, using a distorted scale in a sales chart can make a small increase appear very large, causing management to overestimate sales growth.

# 10. Explain how data visualization supports storytelling and business intelligence in organizations.
# Answer:
# Data visualization converts complex business data into understandable graphical stories. It helps organizations identify trends, communicate findings, compare performance, and make informed business decisions.