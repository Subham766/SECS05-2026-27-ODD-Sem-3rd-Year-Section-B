import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("Advertising.csv")
X = df[["TV"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

plt.scatter(X_test, y_test, label="Actual")
plt.plot(X_test, y_pred, label="Regression Line")
plt.xlabel("TV Advertising")
plt.ylabel("Sales")
plt.title("Linear Regression: TV Advertising vs Sales")
plt.legend()
plt.show()

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# Questions and Answers
# Q1. What is Predictive Analytics, and how is it used in real-world applications?
# Predictive Analytics uses historical data, statistical methods and machine-learning techniques to estimate future outcomes. It is used for sales forecasting, demand prediction, risk analysis, customer behavior prediction and other business decisions.
# Q2. Explain the working principle of the Linear Regression algorithm.
# Linear Regression models the relationship between an independent variable and a continuous dependent variable by fitting the best line to the observed data. The basic equation is Y = b0 + b1X. The coefficients are learned from training data and then used to predict new values.
# Q3. Differentiate between dependent and independent variables with suitable examples.
# The independent variable is the input or predictor used by the model, while the dependent variable is the target being predicted. In an advertising example, TV advertising expenditure is independent and Sales is dependent.
# Q4. Why is it necessary to split the dataset into training and testing sets?
# Splitting the data allows the model to learn from the training set and then be evaluated on unseen testing data. This gives a better estimate of how well the model will generalize to new observations.
# Q5. What is the significance of the R² Score in regression analysis?
# R² indicates the proportion of variation in the dependent variable explained by the regression model. A value closer to 1 generally indicates a stronger fit.
# Q6. Differentiate between MAE, MSE, and RMSE. Which metric is more sensitive to large prediction errors?
# MAE is the mean absolute prediction error. MSE is the mean squared error and penalizes large errors more strongly. RMSE is the square root of MSE and is expressed in the same units as the target. MSE and RMSE are more sensitive to large errors.
# Q7. What assumptions should be satisfied before applying Linear Regression?
# Important assumptions include a reasonably linear relationship, independent observations, approximately constant error variance, and suitably behaved residuals. With multiple predictors, serious multicollinearity should also be avoided.
# Q8. How can overfitting and underfitting affect the performance of a regression model?
# Overfitting occurs when a model learns noise in the training data and performs poorly on unseen data. Underfitting occurs when the model is too simple to capture important relationships. Appropriate features, validation and model complexity help reduce these problems.
# Q9. Mention any three real-world applications of Linear Regression in business or industry.
# Examples include house-price prediction, sales forecasting and product-demand prediction.
# Q10. How can feature selection improve the accuracy and interpretability of a predictive model?
# Feature selection removes irrelevant or redundant variables and retains useful predictors. It can reduce noise and complexity, improve generalization, and make the model easier to interpret.

