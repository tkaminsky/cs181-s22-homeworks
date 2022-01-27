#####################
# CS 181, Spring 2022
# Homework 1, Problem 1
# STARTER CODE
##################

import numpy as np

data = [(0., 0.),
        (1., 0.5),
        (2., 1.),
        (3., 2.),
        (4., 1.),
        (6., 1.5),
        (8., 0.5)]

def compute_loss(tau):
    # TODO
    loss = 0
    return loss

for tau in (0.01, 2, 100):
    print("Loss for tau = " + str(tau) + ": " + str(compute_loss(tau)))