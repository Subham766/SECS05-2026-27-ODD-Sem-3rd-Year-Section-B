import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD DATASET
df = pd.read_csv("netflix_titles.csv")

print("=" * 70)
print("NETFLIX DATASET - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# 2. DISPLAY FIRST RECORDS
print("\nFirst 5 Records:")
print(df.head())

# 3. DATASET DIMENSIONS
print("\nDataset Dimensions:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 4. COLUMN NAMES
print("\nColumn Names:")
print(df.columns.tolist())

# 5. DATA TYPES AND INFORMATION
print("\nDataset Information:")
df.info()

# 6. SUMMARY STATISTICS
print("\nSummary Statistics:")
print(df.describe(include="all"))

# 7. CHECK MISSING VALUES
print("\nMissing Values:")
print(df.isnull().sum())

# 8. VALUE COUNTS
if "type" in df.columns:
    print("\nMovies and TV Shows:")
    print(df["type"].value_counts())

if "rating" in df.columns:
    print("\nContent Ratings:")
    print(df["rating"].value_counts().head(10))

if "release_year" in df.columns:
    print("\nTop Release Years:")
    print(df["release_year"].value_counts().head(10))

# 9. UNIVARIATE ANALYSIS - DISTRIBUTION OF RELEASE YEAR
if "release_year" in df.columns:
    plt.figure(figsize=(10, 5))

    sns.histplot(
        df["release_year"].dropna(),
        bins=30,
        kde=True
    )

    plt.title("Distribution of Release Years")
    plt.xlabel("Release Year")
    plt.ylabel("Number of Titles")

    plt.show()

# 10. CATEGORICAL VARIABLE ANALYSIS - MOVIES VS TV SHOWS
if "type" in df.columns:
    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=df,
        x="type"
    )

    plt.title("Movies vs TV Shows")
    plt.xlabel("Type")
    plt.ylabel("Number of Titles")

    plt.show()

# 11. TOP CONTENT RATINGS
if "rating" in df.columns:
    plt.figure(figsize=(10, 5))

    rating_counts = df["rating"].value_counts().head(10)

    sns.barplot(
        x=rating_counts.index,
        y=rating_counts.values
    )

    plt.title("Top Content Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Number of Titles")

    plt.xticks(rotation=45)

    plt.show()

# 12. TOP COUNTRIES
if "country" in df.columns:
    country_counts = (
        df["country"]
        .dropna()
        .str.split(", ")
        .explode()
        .value_counts()
        .head(10)
    )

    plt.figure(figsize=(10, 5))

    sns.barplot(
        x=country_counts.values,
        y=country_counts.index
    )

    plt.title("Top 10 Countries by Number of Titles")
    plt.xlabel("Number of Titles")
    plt.ylabel("Country")

    plt.show()

# 13. CORRELATION ANALYSIS
numeric_columns = df.select_dtypes(
    include=np.number
).columns

print("\nNumerical Columns:")
print(numeric_columns.tolist())

if len(numeric_columns) >= 2:
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

# 14. BOX PLOT
if "release_year" in df.columns:
    plt.figure(figsize=(8, 5))

    sns.boxplot(
        x=df["release_year"].dropna()
    )

    plt.title("Box Plot of Release Year")
    plt.xlabel("Release Year")

    plt.show()

# 15. SCATTER PLOT
if "release_year" in df.columns:
    yearly_counts = (
        df["release_year"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    yearly_counts.columns = [
        "release_year",
        "number_of_titles"
    ]

    plt.figure(figsize=(10, 5))

    sns.scatterplot(
        data=yearly_counts,
        x="release_year",
        y="number_of_titles"
    )

    plt.title("Release Year vs Number of Titles")
    plt.xlabel("Release Year")
    plt.ylabel("Number of Titles")

    plt.show()

# 16. BIVARIATE ANALYSIS - TYPE VS RATING
if "type" in df.columns and "rating" in df.columns:
    plt.figure(figsize=(12, 6))

    sns.countplot(
        data=df,
        x="rating",
        hue="type",
        order=df["rating"].value_counts().head(10).index
    )

    plt.title("Content Type by Rating")
    plt.xlabel("Rating")
    plt.ylabel("Number of Titles")

    plt.xticks(rotation=45)

    plt.show()

# 17. MULTIVARIATE ANALYSIS
if "type" in df.columns and "release_year" in df.columns:
    yearly_type = (
        df.groupby(
            ["release_year", "type"]
        )
        .size()
        .reset_index(name="count")
    )

    plt.figure(figsize=(12, 6))

    sns.lineplot(
        data=yearly_type,
        x="release_year",
        y="count",
        hue="type"
    )

    plt.title("Content Type Trends Over Release Years")
    plt.xlabel("Release Year")
    plt.ylabel("Number of Titles")

    plt.show()

# 18. TOP GENRES
if "listed_in" in df.columns:
    genre_counts = (
        df["listed_in"]
        .dropna()
        .str.split(", ")
        .explode()
        .value_counts()
        .head(10)
    )

    print("\nTop 10 Genres:")
    print(genre_counts)

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x=genre_counts.values,
        y=genre_counts.index
    )

    plt.title("Top 10 Genres")
    plt.xlabel("Number of Titles")
    plt.ylabel("Genre")

    plt.show()

# 19. FINAL OBSERVATIONS
print("\n" + "=" * 70)
print("KEY OBSERVATIONS")
print("=" * 70)

if "type" in df.columns:
    type_counts = df["type"].value_counts()

    print("\n1. Most common content type:")
    print(type_counts.idxmax())

if "rating" in df.columns:
    print("\n2. Most common content rating:")
    print(df["rating"].mode()[0])

if "release_year" in df.columns:
    print("\n3. Latest release year:")
    print(df["release_year"].max())

    print("\n4. Earliest release year:")
    print(df["release_year"].min())

if "listed_in" in df.columns:
    print("\n5. Most common genre:")
    print(
        df["listed_in"]
        .dropna()
        .str.split(", ")
        .explode()
        .value_counts()
        .idxmax()
    )

print("\nEDA completed successfully!")


# Q1. What is Exploratory Data Analysis (EDA), and why is it performed before machine learning?
# Answer:
# Exploratory Data Analysis (EDA) is the process of examining and understanding a dataset using statistical methods and visualizations. It is performed before machine learning to identify patterns, trends, missing values, outliers, relationships between variables, and data-quality problems. EDA helps in selecting useful features and preparing the data for further analysis or model building.

# Q2. Differentiate between univariate, bivariate, and multivariate analysis with suitable examples.
# Answer:
# •	Univariate Analysis: It analyzes only one variable at a time. Example: analyzing the distribution of release_year. 
# •	Bivariate Analysis: It studies the relationship between two variables. Example: analyzing type and rating. 
# •	Multivariate Analysis: It studies three or more variables together. Example: analyzing release_year, type, and number of titles. 
# These analyses help understand individual variables as well as relationships and patterns in the dataset.

# Q3. What insights can be obtained from a correlation heatmap?
# Answer:
# A correlation heatmap shows the strength and direction of relationships between numerical variables. A value close to +1 indicates a strong positive relationship, while a value close to -1 indicates a strong negative relationship. A value near 0 indicates little or no linear relationship. It helps identify highly related or redundant features.

# Q4. Explain the purpose of histograms, box plots, and scatter plots in EDA.
# Answer:
# •	Histogram: Shows the distribution and frequency of numerical data and helps identify its shape and spread. 
# •	Box Plot: Shows the median, quartiles, range, and possible outliers in numerical data. 
# •	Scatter Plot: Shows the relationship between two numerical variables and helps identify trends or patterns. 
# These visualizations make it easier to understand the characteristics of a dataset.

# Q5. How can EDA help identify data quality issues before analysis?
# Answer:
# EDA helps identify data-quality problems such as missing values, duplicate records, incorrect data types, unusual values, and outliers. Functions such as isnull().sum(), info(), and describe() help inspect the dataset. Visualizations such as box plots and histograms can also reveal unusual distributions and possible errors.

# Q6. Why is correlation important in predictive analytics? Can correlation imply causation?
# Answer:
# Correlation is important because it helps identify relationships between variables and can help in feature selection for predictive models. However, correlation does not imply causation. Two variables may be related without one directly causing the other. Another factor may be responsible for the observed relationship.

# Q7. Which visualization would you use to analyze categorical and numerical variables? Justify your choice.
# Answer:
# For categorical variables, bar charts and count plots are useful because they show the frequency of different categories. For numerical variables, histograms and box plots are useful because they show the distribution, spread, and possible outliers. For relationships between numerical variables, a scatter plot is appropriate.

# Q8. What business insights can be derived from the Netflix dataset through EDA?
# Answer:
# EDA on the Netflix dataset can provide insights about the number of Movies and TV Shows, popular content ratings, major content-producing countries, common genres, and release-year trends. These insights can help understand content preferences and support decisions related to content production, acquisition, and recommendation strategies.


# Q9. How does EDA contribute to feature selection and model building?
# Answer:
# EDA helps identify important and relevant features by studying their distributions and relationships with other variables. It can also identify highly correlated or redundant features, outliers, and missing values. This information helps select better features and prepare cleaner data for machine learning models.

# Q10. What challenges might arise while performing EDA on large-scale real-world datasets?
# Answer:
# Large datasets may contain millions of records and many features, making them difficult to process and visualize. Common challenges include high memory usage, missing values, duplicate records, inconsistent data, outliers, and longer processing time. Large datasets may also contain complex relationships that are difficult to interpret.


