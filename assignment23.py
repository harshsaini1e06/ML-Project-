import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('laptopPrice.csv')
# print(data.head())
print(data.isnull().sum())
# print(data.describe())
sns.histplot(data['price'], kde=True)
plt.title('Distribution of Price')
plt.show()
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
X = data.drop(columns=['price'])
Y = data['price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
linear_regressor = LinearRegression()
decision_tree_regressor = DecisionTreeRegressor(random_state=42)
random_forest_regressor = RandomForestRegressor(random_state=42)
linear_regressor.fit(X_train, Y_train)
Y_pred_lr = linear_regressor.predict(X_test)
print("Linear Regression Performance")
print("Mean Squared Error:", mean_squared_error(Y_test, Y_pred_lr))
print("R^2 Score:", r2_score(Y_test, Y_pred_lr))
decision_tree_regressor.fit(X_train, Y_train)
Y_pred_dt = decision_tree_regressor.predict(X_test)
print("Decision Tree Performance")
print("Mean Squared Error:", mean_squared_error(Y_test, Y_pred_dt))
print("R^2 Score:", r2_score(Y_test, Y_pred_dt))
random_forest_regressor.fit(X_train, Y_train)
Y_pred_rf = random_forest_regressor.predict(X_test)
print("Random Forest Performance")
print("Mean Squared Error:", mean_squared_error(Y_test, Y_pred_rf))
print("R^2 Score:", r2_score(Y_test, Y_pred_rf))
importances = random_forest_regressor.feature_importances_
features = X.columns
indices = np.argsort(importances)
plt.figure(figsize=(10, 8))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')