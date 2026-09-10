import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, ConfusionMatrixDisplay
)

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Convert TotalCharges to numeric and handle missing values
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# Encode categorical variables
X = df.drop(columns=["customerID", "Churn"])
X = pd.get_dummies(X, drop_first=True)
y = LabelEncoder().fit_transform(df["Churn"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = DecisionTreeClassifier(
    criterion="gini", max_depth=5, random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix")
plt.show()

plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No Churn", "Churn"],
    filled=True,
    max_depth=3
)
plt.title("Decision Tree")
plt.show()

importance = pd.Series(model.feature_importances_, index=X.columns)
print("\nTop influential features:")
print(importance.sort_values(ascending=False).head(10))

# Questions and Answers
# Q1. What is classification, and how does it differ from regression?
# Classification predicts discrete class labels, while regression predicts continuous numerical values.
# Q2. Explain the working principle of the Decision Tree algorithm.
# A Decision Tree recursively splits the data using feature-based rules chosen to make the resulting groups more homogeneous with respect to the target class. Splitting continues until stopping criteria are reached, and a leaf provides the predicted class.
# Q3. Differentiate between Gini Index and Entropy as splitting criteria.
# Both measure impurity. Gini Index is based on the probability of misclassification, while Entropy measures information uncertainty. Lower impurity indicates a purer split.
# Q4. What is a Confusion Matrix? Explain its components.
# A confusion matrix summarizes classification results using True Positives, True Negatives, False Positives and False Negatives.
# Q5. Define Accuracy, Precision, Recall, and F1-Score. Why are these metrics important?
# Accuracy is the proportion of all correct predictions. Precision is the proportion of predicted positives that are correct. Recall is the proportion of actual positives detected. F1-Score is the harmonic mean of precision and recall. Together they provide a more complete view than accuracy alone.
# Q6. What is overfitting in a Decision Tree? How can it be reduced?
# Overfitting occurs when the tree becomes too complex and models noise in the training data. It can be reduced using max_depth, min_samples_split, min_samples_leaf, pruning, and validation.
# Q7. Why is customer churn prediction important for businesses?
# Early identification of customers likely to leave enables targeted retention actions, improves customer satisfaction and can reduce revenue loss.
# Q8. What are the advantages and limitations of the Decision Tree algorithm?
# Advantages include interpretability, simple rule-based decisions, little need for feature scaling, and support for nonlinear relationships. Limitations include overfitting, instability to small data changes, and potentially poorer generalization than ensembles.
# Q9. Mention three real-world applications of Decision Tree Classification other than customer churn prediction.
# Credit-risk assessment, medical diagnosis, and fraud detection are three examples.
# Q10. How can the insights obtained from a churn prediction model help organizations improve customer retention and business profitability?
# Organizations can identify high-risk customers, understand the features associated with churn, prioritize retention campaigns, personalize offers and measure which interventions reduce churn, thereby protecting revenue.

