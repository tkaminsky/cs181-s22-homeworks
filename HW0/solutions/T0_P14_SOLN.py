import numpy as np
np.random.seed(181)

# Item 1
n = 20
x = np.random.rand(n) * 20 - 10
y = np.random.rand(n) * 60 + 20

# Item 2
# TODO

# Item 3
print("The answer to 14.3 is...")
z = np.zeros(n)
for i in range(n):
    z[i] = (y[i] + 10) * x[i]/5
mean_z = np.mean(z)
std_z = np.std(z)
print(mean_z, std_z)

# Item 4
print("The answer to 14.4 is...")
argmax_y = np.argmax(y)
print(x[argmax_y], y[argmax_y])

# Item 5
print("The answer to 14.5 is...")
print(sum(y[x > 0]))