#####################
# CS 181, Spring 2022
# Homework 1, Problem 4
# Start Code
##################

import csv
import numpy as np
import matplotlib.pyplot as plt

csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985

# Plot the data.
plt.figure(1)
plt.plot(years, republican_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.figure(2)
plt.plot(years, sunspot_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Sunspots")
plt.figure(3)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()

# Create the simplest basis, with just the time and an offset.
X = np.vstack((np.ones(years.shape), years)).T

# TODO: basis functions
# Based on the letter input for part ('a','b','c','d'), output numpy arrays for the bases.
# The shape of arrays you return should be: (a) 24x6, (b) 24x12, (c) 24x6, (c) 24x26
# xx is the input of years (or any variable you want to turn into the appropriate basis).
# is_years is a Boolean variable which indicates whether or not the input variable is
# years; if so, is_years should be True, and if the input varible is sunspots, is_years
# should be false

mu_js = np.array(range(1960, 2010, 15))

def standardize(x, mu):
    return np.exp((-1 * (x - mu) ** 2 ) / 25)

def make_basis(xx,part='a',is_years=True):
#DO NOT CHANGE LINES 65-69

    if part == 'a' and is_years:
        xx = (xx - np.array([1960]*len(xx)))/40
        
    if part == "a" and not is_years:
        xx = xx/20
        
    
    if part == "a":
        return np.array([[x**i for i in range(6)] for x in xx])
    
    if part == "b":
        basis = np.array([[standardize(x, i) for i in range(1960, 2011, 15)] for x in xx])
        n,m = basis.shape
        return np.hstack((np.ones((n,1)), basis))
    
    if part == "c":
        basis = np.array([[np.cos(x / j) for j in range(1, 6, 1)] for x in xx])
        n,m = basis.shape
        return np.hstack((np.ones((n,1)), basis))
    
    if part == "d":
        basis = np.array([[np.cos(x / j) for j in range(1, 26, 1)] for x in xx])
        n,m = basis.shape
        return np.hstack((np.ones((n,1)), basis))

    return X

# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
def find_weights(X,Y):
    w = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
    return w

# Part a coordinates generation

basis_a = make_basis(years, part="c")
weights = find_weights(basis_a, Y)

domain = np.array(range(1960, 2020, 1))
domain_basis = make_basis(domain, part="c")

image = np.array([np.dot(weights, el) for el in domain_basis])
print(domain.shape)
print(image.shape)



w = find_weights(X,Y)

plt.plot(domain, image, color="magenta")

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)
grid_X = np.vstack((np.ones(grid_years.shape), grid_years))
grid_Yhat  = np.dot(grid_X.T, w)

# TODO: plot and report sum of squared error for each basis
make_basis(years)


# Plot the data and the regression line.
plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.show()