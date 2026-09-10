import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("Mall_Customers.csv")
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
sil_scores = []
ks = range(2, 11)

for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

plt.plot(list(ks), inertias, marker="o")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

plt.plot(list(ks), sil_scores, marker="o")
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score")
plt.show()

k = 5
model = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(X_scaled)

plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"], c=df["Cluster"])
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments using K-Means")
plt.show()

print(df.groupby("Cluster")[["Annual Income (k$)", "Spending Score (1-100)"]].mean())

# Questions and Answers
# Q1. What is clustering? How does it differ from classification?
# Clustering groups similar observations without predefined labels, while classification learns from labeled data to assign observations to known classes.
# Q2. Explain the working principle of the K-Means Clustering algorithm.
# K-Means chooses K centroids, assigns each observation to its nearest centroid, recalculates centroids from the assigned points, and repeats until the assignments stabilize or the stopping condition is reached.
# Q3. Why is feature scaling important before applying K-Means clustering?
# K-Means uses distance calculations. Features with larger numerical scales can dominate those with smaller scales, so standardization or normalization helps each feature contribute more fairly.
# Q4. What is the Elbow Method, and how does it help determine the optimal number of clusters?
# The Elbow Method plots within-cluster sum of squares (inertia) for different K values. The point where the improvement starts becoming much smaller is used as a practical choice for K.
# Q5. What is the Silhouette Score? How is it used to evaluate clustering performance?
# The Silhouette Score measures how close points are to their own cluster compared with other clusters. Higher values generally indicate better-separated and more cohesive clusters.
# Q6. Why is K-Means considered an unsupervised machine learning algorithm?
# K-Means works without a target label supplied in advance. It discovers groups from the structure of the input data.
# Q7. Mention any four real-world applications of customer segmentation.
# Targeted marketing, personalized product recommendations, customer loyalty programs, and differentiated pricing or offers.
# Q8. What are the limitations of the K-Means algorithm?
# The number of clusters K must be selected, results can depend on initialization, it is sensitive to scale and outliers, and it works best when clusters are reasonably compact and separated.
# Q9. How can businesses use customer segmentation to improve marketing and customer retention?
# Businesses can identify groups with different needs and spending patterns, then create targeted offers, personalized communication and retention strategies for each segment.
# Q10. Compare K-Means Clustering with Hierarchical Clustering based on their working principles and applications.
# K-Means directly partitions data into a chosen number of clusters by repeatedly updating centroids. Hierarchical clustering builds a hierarchy of nested clusters that can be viewed as a dendrogram. K-Means is often efficient for larger datasets, while hierarchical clustering is useful when the nested relationships among groups are important.

