# EXPERIMENT 9
# Data Storytelling and Business Insight Generation
# Using Superstore Sales Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD DATASET

df = pd.read_csv("Superstore_Sales.csv")

print("Dataset loaded successfully!")
print("\nFirst 5 rows:")
print(df.head())

# 2. BASIC INFORMATION

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# 3. REMOVE DUPLICATES

print("\nDuplicate Rows:", df.duplicated().sum())

df = df.drop_duplicates()

# 4. KPI CALCULATION

sales = df["Sales"].sum()
profit = df["Profit"].sum()

print("\n========== KEY PERFORMANCE INDICATORS ==========")
print("Total Sales  :", round(sales, 2))
print("Total Profit :", round(profit, 2))
print("Number of Orders:", df.shape[0])

if "Quantity" in df.columns:
    quantity = df["Quantity"].sum()
    print("Total Quantity:", quantity)

# 5. SALES BY CATEGORY

category_sales = df.groupby("Category")["Sales"].sum().sort_values(
    ascending=False
)

print("\nSales by Category:")
print(category_sales)

plt.figure(figsize=(8, 5))
category_sales.plot(kind="bar")
plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 6. PROFIT BY CATEGORY

category_profit = df.groupby("Category")["Profit"].sum().sort_values(
    ascending=False
)

print("\nProfit by Category:")
print(category_profit)

plt.figure(figsize=(8, 5))
category_profit.plot(kind="bar")
plt.title("Profit by Category")
plt.xlabel("Category")
plt.ylabel("Profit")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 7. SALES BY REGION

if "Region" in df.columns:

    region_sales = df.groupby("Region")["Sales"].sum().sort_values(
        ascending=False
    )

    print("\nSales by Region:")
    print(region_sales)

    plt.figure(figsize=(8, 5))
    region_sales.plot(kind="bar")
    plt.title("Sales by Region")
    plt.xlabel("Region")
    plt.ylabel("Sales")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# 8. PROFIT BY REGION

if "Region" in df.columns:

    region_profit = df.groupby("Region")["Profit"].sum().sort_values(
        ascending=False
    )

    print("\nProfit by Region:")
    print(region_profit)

    plt.figure(figsize=(8, 5))
    region_profit.plot(kind="bar")
    plt.title("Profit by Region")
    plt.xlabel("Region")
    plt.ylabel("Profit")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# 9. SALES BY SUB-CATEGORY

if "Sub-Category" in df.columns:

    subcategory_sales = df.groupby("Sub-Category")["Sales"].sum().sort_values(
        ascending=False
    )

    print("\nTop Sub-Categories:")
    print(subcategory_sales.head(10))

    plt.figure(figsize=(10, 6))
    subcategory_sales.head(10).plot(kind="bar")
    plt.title("Top 10 Sub-Categories by Sales")
    plt.xlabel("Sub-Category")
    plt.ylabel("Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 10. SALES TREND OVER TIME

if "Order Date" in df.columns:

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        errors="coerce"
    )

    monthly_sales = df.groupby(
        df["Order Date"].dt.to_period("M")
    )["Sales"].sum()

    print("\nMonthly Sales:")
    print(monthly_sales)

    plt.figure(figsize=(12, 5))
    monthly_sales.plot(kind="line", marker="o")

    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 11. PROFIT vs SALES

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Sales",
    y="Profit"
)

plt.title("Sales vs Profit")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.tight_layout()
plt.show()

# 12. CORRELATION HEATMAP

numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(10, 6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
# 13. TOP 10 PRODUCTS

if "Product Name" in df.columns:

    top_products = df.groupby("Product Name")["Sales"].sum().sort_values(
        ascending=False
    ).head(10)

    print("\nTop 10 Products:")
    print(top_products)

    plt.figure(figsize=(10, 6))
    top_products.plot(kind="bar")

    plt.title("Top 10 Products by Sales")
    plt.xlabel("Product")
    plt.ylabel("Sales")
    plt.xticks(rotation=75)

    plt.tight_layout()
    plt.show()

# 14. BUSINESS INSIGHTS

print("\n========== BUSINESS INSIGHTS ==========")
print("1. Category-wise sales can be used to identify the strongest product category.")
print("2. Region-wise sales helps identify high-performing and low-performing regions.")
print("3. Monthly sales trends help identify growth patterns and seasonal changes.")
print("4. Profit analysis helps determine whether high sales are also generating good profits.")
print("5. Top-product analysis helps identify products that contribute significantly to revenue.")
print("6. Correlation analysis helps understand relationships between numerical variables.")

# 15. RECOMMENDATIONS

print("\n========== RECOMMENDATIONS ==========")
print("1. Focus marketing activities on high-performing categories.")
print("2. Analyze low-profit categories and review pricing or discount strategies.")
print("3. Improve sales strategies in low-performing regions.")
print("4. Maintain sufficient inventory for high-demand products.")
print("5. Use monthly sales trends for better business planning.")
print("\nExperiment 9 completed successfully!")

# Q1. What is data storytelling? How is it different from data visualization
# Answer:
# Data storytelling is the process of combining data, visualizations, and narrative to explain important findings clearly. Data visualization mainly represents data through charts and graphs, while data storytelling explains the meaning of those visualizations and connects them to business decisions.

# Q2. Why is storytelling important in business analytics and decision-making?
# Answer:
# Storytelling makes complex analytical results easier to understand. It helps stakeholders identify important trends, problems, and opportunities and supports better business decisions by connecting data with practical actions.

# Q3. What are the essential components of an effective data story?
# Answer:
# The main components are data, visualizations, narrative, context, insights, and recommendations. Data provides evidence, visualizations make patterns easier to understand, and the narrative explains the meaning and suggests suitable actions.

# Q4. How do Key Performance Indicators (KPIs) enhance business reporting?
# Answer:
# KPIs provide measurable values that show how well a business is performing. They help managers monitor important areas such as sales, profit, revenue, customer satisfaction, and growth and identify performance gaps quickly.

# Q5. Why should visualizations be arranged in a logical sequence while presenting analytical findings?
# Answer:
# A logical sequence helps the audience understand the analysis step by step. It creates a clear flow from the problem and data to findings, insights, and recommendations, making the overall data story easier to follow.

# Q6. What factors should be considered while selecting visualizations for a business presentation?
# Answer:
# The type of data, purpose of analysis, audience, number of variables, readability, and message that needs to be communicated should be considered. For example, bar charts are useful for comparisons, while line charts are useful for showing trends over time.

# Q7. Explain how dashboards and storytelling complement each other in Business Intelligence.
# Answer:
# Dashboards provide an interactive view of important data and KPIs, while storytelling explains what the data means. Together, they allow users to explore information and understand the important insights needed for decision-making.

# Q8. What challenges may arise while communicating analytical insights to non-technical stakeholders?
# Answer:
# Non-technical stakeholders may have difficulty understanding complex statistical terms, technical models, or complicated charts. Too much information can also cause confusion. Therefore, analysts should use simple language, clear visualizations, and business-focused explanations.

# Q9. Give two real-world examples where data storytelling has influenced business or policy decisions.
# Answer:
# Example 1: Retail companies can analyze sales and customer data to identify popular products and adjust inventory and marketing strategies.
# Example 2: Governments can use COVID-19 statistics and visual dashboards to communicate infection trends and support decisions related to public health measures.

# Q10. How can effective data storytelling improve strategic planning and organizational performance?
# Answer:
# Effective data storytelling helps organizations understand their current performance, identify trends and risks, and recognize opportunities. It supports evidence-based planning and helps managers make informed decisions that can improve organizational performance.