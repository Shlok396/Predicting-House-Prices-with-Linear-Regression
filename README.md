PROJECT  TITLE  :  Predicting House Prices Using Linear Regression

OBJECTIVE
The objective of this project is to build a predictive model using Linear Regression to estimate house prices based on relevant numerical features from the given dataset.
This project aims to understand how different features influence house prices and to evaluate the performance of a linear regression model using appropriate metrics.

 STEPS PERFORMED
1. Data Collection
The dataset was provided in CSV format and uploaded to Google Colab.
It contains various numerical features related to houses along with a target variable representing house price.

2. Data Exploration
The dataset was loaded using the Pandas library.
Basic exploration was performed:
Viewing the first few rows of the dataset
Checking column names and data types
Understanding the target variable and input features

3. Data Cleaning
Checked for missing or null values.
Handled missing data (if any) using
Removal of rows with missing values or
Filling missing values using suitable methods.
Ensured all selected features were numerical for linear regression.

4. Feature Selection
Identified important numerical features that influence house prices.
Selected independent variables (features) and the dependent variable (house price).
Removed unnecessary or irrelevant columns.

5. Data Splitting
The dataset was split into:
Training set
Testing set
This helps in evaluating model performance on unseen data.

6. Model Training
Implemented Linear Regression using the Scikit-Learn library.
Trained the model using the training dataset.
The model learned the relationship between features and house prices.

7. Model Evaluation
Predictions were made on the test dataset.
Model performance was evaluated using:
Mean Squared Error (MSE)
R-squared (R²) score
These metrics indicate how well the model predicts house prices.

8. Visualization
Plotted graphs to compare:
Actual house prices
Predicted house prices
Visualization helped in understanding model accuracy and error patterns.

TOOLS USED
1.	Python
2.	Google Colab
3.	Pandas – Data handling and analysis
4.	NumPy – Numerical operations
5.	Scikit-Learn – Linear regression model and evaluation
6.	Matplotlib / Seaborn – Data visualization




CODE:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files
uploaded=files.upload()
print(uploaded

df = pd.read_csv(list(uploaded.keys())[0])
df.head()

df.shape
df.info()
df.describe()

df.isnull().sum()
df = df.dropna()
df.columns 

X = df.drop('price', axis=1)
y = df['price']

X = pd.get_dummies(X, drop_first=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R-squared Score:", r2)

comparison = pd.DataFrame({
    'Actual Price': y_test,
    'Predicted Price': y_pred
})comparison.head()

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

coefficients = pd.DataFrame(
    model.coef_,
    X.columns,
   columns=['Coefficient']


THE OUTCOME IN BRIEF
1.	A linear regression model was successfully built to predict house prices.
2.	The model was trained and tested on the given dataset.
3.	Evaluation metrics showed how accurately the model predicts house prices. 

 

 
