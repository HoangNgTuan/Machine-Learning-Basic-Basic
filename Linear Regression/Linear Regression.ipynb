from __future__ import division, print_function, unicode-literals # fit with both py2 and py3
import numpy as np
import matplotlib.pyplot as plt  #To draw graph

#height (cm)
x = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T # T: horizontal array
#weight (kg)
y = np.array([[49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

#Add data to form a graph
plt.plot(x, y, 'ro')
plt.axis([140, 190, 45, 75]) # On the graph, Height will be shown from 140 - 190, Weight: 45 - 75
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()   # Show the chart, we will see our data barely form a line
             # => Linear function: Weight = w1*Height + bias
  
#Building Xbar (X with a line) 
one = np.ones((x.shape[0], 1)) # x.shape[0]: The first element of x's shape  
Xbar = np.concatenate((one, x), axis = 1) #Combine array 'one' to array 'x'

#Calculating weights of the fitting line    
A = np.dot(Xbar.T, Xbar)    # Xbar.T * Xbar
b = np.dot(Xbar.T, y)     
w = np.dot(np.linalg.pinv(A), b) # pseudo inverse of A* b
print('w = ',w)

#Preparing the fitting line
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(145, 185, 2) # Graph has x = [145; 185]
y0 = w_0 + w_1*x0

#Drawing the fitting line
plt.plot(x.T, y.T, 'ro') #ro stands for red circle, which mean the point (x,y) will be represented by a red circle
plt.plot(x0, y0)          #the fitting line
plt.axis([140, 190, 45, 75]) 
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()

# Up to this point, the graph barely fit with all the red circle (the data point)
# Which mean our training has a good result.

# We can use our training up there to predict upcoming dataset
# Pretending we predict weight of person with height 155 and 160, both datas don't appear in training dataset

#Our linear regression with updated weight and bias
y1 = w_1*155 +w_0
y2 = w_1*160 +w_0

print( u'Predict weight of person with height 155 cm: %.2f (kg), real number: 52 (kg)' %(y1) )
print( u'Predict weight of person with height 160 cm: %.2f (kg), real number: 56 (kg)' %(y2) )

