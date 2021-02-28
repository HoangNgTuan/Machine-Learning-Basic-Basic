# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import math
import numpy as np
import matplotlib.pyplot as plt


# Suppose we have a f(x) function  
# f(x) = x**2 + 5*np.sin(x)        (x^2 + 5sin(x))

def grad(x):
  return 2*x + 5*np.cos(x)    # Gradient of f(x)

def cost(x):
  return x**2 + 5*np.sin(x)   # f(x) function, just for checking if our program is right on cue

def myGD1(eta, x0):        # Gradient Descent algorithm
  x = [x0]                  # Make x a tuple so it can't be modified
  for i in range(100):            # Max iterations is 100 (not important)
    x_new = x[-1] - eta*grad(x[-1])    # From now, x[-1] is the last element, which mean x0
    if abs(grad(x_new)) < 1e-3:     # 1e-3 = 1*10^-3, the smaller grad value, the better
      break                       # When grad value almost equal to local minimum
    x.append(x_new)               # Update tuple x with new value (x_new is x[-1] now)
    return (x, i)               # Return a tuple and number of iterations

(x1, i1) = myGD1(.1, -5)
(x2, i2) = myGD1(.1, 5)
print('Solution x1 = %f, cost = %f, obtained after %d iterations' %(x1[-1], cost(x1[-1]), i1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations' %(x2[-1], cost(x2[-1]), i2))

#The cost from both 2 print has almost the same value, the algorithm went smoothly
