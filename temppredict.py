import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#======================================================================
#Loading the Data
df = pd.read_csv("humidity.csv")

print("Dataset Shape:", df.shape)
df.head()
df.info()
print("-"*30)
print(df.isnull().sum())
print("-"*30)

#Cleaning data
df = df.dropna()
print(df.isnull().sum())
print("-"*30)

#======================================================================

#EDA
#Correlation Heatmap
plt.figure(figsize = (8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

#Humidity vs Temperature
plt.figure(figsize = (6,4))
sns.scatterplot(x=df["humidity"], y=df["temperature"])
plt.title("Humidity vs Temperature")
plt.show()

#Feature Selection
x = df[["humidity", "pressure", "lat", "lon"]]
y = df["temperature"]
#Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
print("Training samples:", x_train.shape)
print("-"*30)
print("Test samples:", x_test.shape)
print("-"*30)

#=====================================================================

#Training the ML model
model = LinearRegression()
model.fit(x_train, y_train)
#Making Predictions
y_pred = model.predict(x_test)
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})
results.head()
print("-"*30)

#Model Evaluation
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)
print("-"*30)

r2 = r2_score(y_test, y_pred)
print("R2:", r2)
print("-"*30)

# Take a small sample for clearer plotting
sample = results.sample(2000)

#Visualization of Predictions
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Actual vs Predicted Temperature")
plt.show()


#The visualized scatter plot compares actual temperature values with the temperatures predicted by the linear regression model.
#Ideally, points should lie close to a diagonal line indicating accurate predictions.
#But, in this model, most points cluster in a region showing a positive relationship between actual and predicted values,
#although some outliers appear due to irregular data points in the dataset.