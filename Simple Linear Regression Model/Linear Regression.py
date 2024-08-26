import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# Read data from the CSV file
data = pd.read_csv('Salary_dataset.csv')

# Extract the relevant columns
x = data['YearsExperience']
y = data['Salary']

# Perform linear regression
slope, intercept, r, p, std_err = stats.linregress(x, y)

# Function to predict y values using the regression line
def myfunc(x):
    return slope * x + intercept

# Generate the regression line model
mymodel = list(map(myfunc, x))

# Plot the scatter plot and the regression line
plt.scatter(x, y)
plt.plot(x, mymodel)
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Years of Experience vs. Salary')
plt.show()

# Print the correlation coefficient (r)
print(f"Correlation coefficient (r): {r}")

# Predict a value for a new x (e.g., x = 10)
predicted_value = myfunc(10)
print(f"Predicted Salary for 10 years of experience: {predicted_value}")
