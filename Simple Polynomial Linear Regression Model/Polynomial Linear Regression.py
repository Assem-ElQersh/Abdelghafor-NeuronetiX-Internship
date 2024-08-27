import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# Load the data
df = pd.read_csv('rank_salary.csv')

# Use 'Level' as the independent variable (x) and 'Salary' as the dependent variable (y)
x = df['Level']
y = df['Salary']

# Create the polynomial model (e.g., degree 3)
mymodel = np.poly1d(np.polyfit(x, y, 3))

# Generate a line for plotting
myline = np.linspace(min(x), max(x), 100)

# Plotting
plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# Calculate and print the R-squared value
r2 = r2_score(y, mymodel(x))
print(f'R-squared: {r2}')