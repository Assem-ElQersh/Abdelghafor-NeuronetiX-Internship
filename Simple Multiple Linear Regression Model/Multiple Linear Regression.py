import pandas as pd
from sklearn import linear_model

# Load the data
df = pd.read_csv('multiple_linear_regression_dataset.csv')

# Use 'age' and 'experience' as independent variables (X) and 'income' as the dependent variable (y)
X = df[['age', 'experience']]
y = df['income']

# Create and train the model
regr = linear_model.LinearRegression()
regr.fit(X, y)

# Print the coefficients
print(f'Coefficients: {regr.coef_}')
