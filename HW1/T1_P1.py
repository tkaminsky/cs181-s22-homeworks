#####################
# CS 181, Spring 2022
# Homework 1, Problem 1
# STARTER CODE
##################

from xml.etree.ElementTree import TreeBuilder
import numpy as np
import matplotlib.pyplot as plt

data = [(0., 0.),
        (1., 0.5),
        (2., 1.),
        (3., 2.),
        (4., 1.),
        (6., 1.5),
        (8., 0.5)]
x = np.array([point[0] for point in data])
y = np.array([point[1] for point in data])

# Kernel Function
def K(x1, x2, tau):
    return np.exp(-1 * np.linalg.norm(x1 - x2) ** 2 / tau)

# Kernel Regressor Function
def f(x_curr, i, tau):
    sum = 0
    for j in range(len(x)):
        if (i != j):
            sum += K(x_curr, x[j], tau) * y[j]
    return sum

# 
def compute_loss(tau):
    loss = 0
    
    for i in range(len(y)):
        loss += (y[i] - f(x[i], i, tau)) ** 2
    return loss

for tau in (0.01, 2, 100):
    print("Loss for tau = " + str(tau) + ": " + str(compute_loss(tau)))

    

# Part 4: Plotting data for each tau

domain = np.arange(0,12.1,.1, dtype=float)

# Lengthscale tau = .01
f1 = [f(i, -1, .01) for i in domain]
# tau = 2
f2 = [f(i, -1, 2) for i in domain]
# tau = 100
f3 = [f(i, -1, 100) for i in domain]

f4 = [f(i, -1, 100000) for i in domain]

plt.figure()

plt.plot(domain,f1,color="red", label="tau = .01")
plt.plot(domain,f2,color="blue", label="tau = 2")
plt.plot(domain,f3,color="green", label="tau = 100")
plt.plot(domain,f4,color="green", label="tau = 100000")

plt.scatter(x,y,color="black", label="data points")

plt.xlabel("Input (x)")
plt.ylabel("f(x)")
plt.title("HW1 Problem 1 pt. 4 Plot")
plt.legend(loc="upper right")

plt.show()

# Part 5: Gradient Descent Solution

# Using some really computationally inefficient stuff :/

# The other part of the derivative (not just f)
def g(x, x_j, tau, i):
    sum = 0
    for j in range(len(x)):
        if (i != j):
            sum += y[j] * (np.linalg.norm(x_j - x[j]) ** 2) * K(x_j, x[j], tau)
    return sum

def loss_prime(tau):
    sum = 0
    for i in range(len(y)):
        sum += (y[i] - f(x[i], i, tau)) * g(x, x[i], tau, i)
    return (-2 / tau) * sum

def gd(th0, epsilon, step):
    th = th0
    th_prev = th
    th -= step * loss_prime(th)
    while (abs(th - th_prev) <= epsilon):
        th_prev = th
        th -= step * loss_prime(th)
    return th
        

print("Trying gradient descent for th0 = 2")
        
tau = gd(2, .0001, .005)
print("I found tau = " + str(tau)) 
