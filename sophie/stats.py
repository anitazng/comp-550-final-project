import pandas as pd
import statsmodels.api as sm

# Load the data from the CSV file
df = pd.read_csv(open('results.tsv', 'r'))

# Define the independent and dependent variables
X = df['similarity']
y = df['baseline']

# Add a constant to the independent variable (required for the regression)
X = sm.add_constant(X)

# Fit the probit model
model = sm.OLS(y, X)
result = model.fit()

# Print the summary of the regression
print(result.summary())