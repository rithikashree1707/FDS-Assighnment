#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 13:34:55 2023

@author: tamadaritikashree
"""
# Import NumPy, a powerful library for numerical operations
import numpy as np
# Import the matplotlib.pyplot as plt, used to create plots and charts
import matplotlib.pyplot as plt
# Import the integrate module from SciPy for numerical integration
from scipy import integrate
# Import the pandas library as pd, used for analysis
import pandas as pd

# Load data into a Pandas DataFrame from a CSV file
df = pd.read_csv('data6-1.csv', header=None, names=['salary'] )
# Get the data from the'salary' column in the DataFrame
salary = df['salary']

# Plot a histogram of salary data
plt.hist(salary, bins=30, density=True, alpha=0.5, color='blue', label='Salary Distribution')
# label for x-axis
plt.xlabel('Salaries')
# label for y-axis
plt.ylabel('Probability')
# Plot the legend
plt.legend()
# show the plot
plt.show()

# Compute the mean salary
mean_salary = np.mean(salary)
# print the mean salary
print(f"Mean annual salary (~W): ${mean_salary:.2f}")

""" Calculate the probability density function (PDF) for a given value 'x' in 
    a salary distribution.

    Parameters:
    x (float or array-like): The value(s) at which to evaluate the PDF.

    Returns:
    float or array-like: The value(s) of the PDF at the input x."""
def pdf(x):
    return 1 / (np.std(salary) * np.sqrt(2 * np.pi)) * np.exp(-(x - np.mean
                                        (salary))**2 / (2 * np.var(salary)))

# Define upper and lower limits for integration
W = mean_salary
lower_limit = W
upper_limit = 1.25 * W

# Integrate to find the fraction of the population within certain salary limits
result, _ = integrate.quad(pdf, lower_limit, upper_limit)
fraction_population = result
# Print the function
print(f"Fraction of population with salaries between ~W and 1.25*~W: {fraction_population:.2f}")
