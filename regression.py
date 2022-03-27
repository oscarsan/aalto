# Import basic libraries needed in this round
import numpy as np 
import pandas as pd  
from matplotlib import pyplot as plt
import pdb
from sklearn.linear_model import LinearRegression
from IPython.display import display, Math
from sklearn.metrics import mean_squared_error
 

def load_housing_data(m=20, n=10):
    # Load dataframe from csv
    df = pd.read_csv("./helsinki_apartment_prices.csv", index_col=0)  
    
    #pdb.set_trace()
    # Extract feature matrix and label vector from dataframe
    X = df.iloc[:m,:n].to_numpy()
    y = df.iloc[:m,-1].to_numpy().reshape(-1)
    
    return X, y
    
# Load the housing data
X, y = load_housing_data(n=2)
print(X.shape, y.shape)

# Create a figure with 3 subplots in 1 row 
#fig, ax = plt.subplots(2, 5, figsize=(15,5))

#pdb.set_trace()

# Create first subplot 
#for i in range(2):
#    for z in range(5):
#        ax[i,z].scatter(X[:,i+z], y)
        # ax[i,z].set_title('first feature $x_{1}$ vs. label $y$')
        # ax[i,z].set_xlabel('$x_{1}$: Average number of rooms')
        # ax[i,z].set_ylabel('$y$: Median apartment value (10000€)')



# Display the figure containing two subplots 
#plt.show()
# Create the linear regression object
reg = LinearRegression(fit_intercept=True) 

# Fit the linear regression model  
reg.fit(X, y)

# Get the optimal weight vector w of the fitted model 
w_opt = reg.coef_

# Get the optimal intercept of the fitted model  
w_intercept = reg.intercept_

# Print the optimal weight vector 
display(Math(r'$\mathbf{w}_{\rm opt} ='))
print(w_opt)

# Print the optimal intercept
display(Math(r'${w}_{0 \rm opt} ='))
print(w_intercept)


# Calculate the predicted labels of the data points in the training set
y_pred = reg.predict(X)

# Calculate the MSE of the true and predicted labels of the training set
training_error = mean_squared_error(y, y_pred)

# Print training error 
print("\nThe resulting mean squared error (training error) is ", training_error)


###FIRST TASK

max_r = 10

# Load the dataset using 10 features 
X, y = load_housing_data(n=10)
  
# Vector for storing the training error for each r
linreg_error = np.zeros(max_r)    


for i in range(1,11):
### STUDENT TASK ###

  reg = LinearRegression(fit_intercept=True) 
  reg.fit(X[:,range(i)], y)
  w_opt = reg.coef_
  w_intercept = reg.intercept_
  y_pred = reg.predict(X[:,range(i)])
  # Calculate the MSE of the true and predicted labels of the training   
  linreg_error[i-1]= mean_squared_error(y, y_pred)


print(linreg_error)

assert linreg_error.shape == (max_r,), "'linreg_error' has the wrong shape."
assert linreg_error[9] < 0.8 * linreg_error[2], "training errors are not correct"
assert linreg_error[5] > linreg_error[6], "training errors are not correct"

print('Sanity check passed!')


# Print the training errors
print(f"Training errors (rounded to 2 decimals): \n {np.round(linreg_error, 2)}")

# # create a numpy array "r_values" containing the values 1,2...,max_r 
# r_values = np.linspace(1, max_r, max_r, endpoint=True)
# # create a plot object which can be accessed using variables "fig" and "axes"
# fig, axes = plt.subplots(1,1, figsize=(8, 5))
# # add a curve representing the average squared error for each choice of r 
# axes.plot(r_values, linreg_error, label='MSE', color='red')
# # add captions for the horizontal and vertical axes 
# axes.set_xlabel('features')
# axes.set_ylabel('empirical error')
# # add a title to the plot 
# axes.set_title('Training error vs number of features')
# axes.legend()
# plt.tight_layout()
# plt.show()


max_m = 10     

# Load the dataset using n=2 features 
X, y = load_housing_data(n=2)   

# Array in which to store the training errors of the different number of data points
train_error = np.zeros(max_m)     



for i in range(1,11):

  reg = LinearRegression(fit_intercept=True) 
  reg.fit(X[range(i),:], y[range(i)])
  w_opt = reg.coef_
  w_intercept = reg.intercept_
  y_pred = reg.predict(X[range(i),:])
  # Calculate the MSE of the true and predicted labels of the training   
  train_error[i-1]= mean_squared_error(y[range(i)], y_pred)


# Perform sanity checks on the results
assert train_error.shape == (10,), "'train_error' has wrong dimensions."
assert train_error[0] < 100 * train_error[3], "training errors are not correct"
assert train_error[2] > train_error[1], "training errors are not correct"

print('Sanity checks passed!')


def load_corrupted_data():
    X, y = load_housing_data(n=1)
    
    # perturb the label of the data point with lowest x_1
    y[np.argmin(X)] = 80
    
    return X, y

X, y = load_housing_data(n=1)
X_cor, y_cor = load_corrupted_data()

# # Plot the real and the corrupted datasets for comparison

# fig, ax = plt.subplots(1, 2,  figsize=(13,6))   # create a figure with two horizontal subplots
# ax[0].scatter(X, y)
# ax[0].set_xlabel('$x_1$: Average number of rooms')
# ax[0].set_ylabel("$y$: Median apartment value (10000€)")
# ax[0].set_title("real-estate dataset $\mathcal{D}$")
# ax[1].scatter(X_cor, y_cor)
# ax[1].set_xlabel('$x_1$: Average number of rooms')
# ax[1].set_ylabel("$y$: Median apartment value (10000€)")
# ax[1].set_title("corrupted dataset $\mathcal{D}'$")
# plt.show()


# learn a linear predictor map by minimizing MSE incurred on the dataset D
reg = LinearRegression(fit_intercept=True) 
reg = reg.fit(X, y)                       
y_pred = reg.predict(X)                   

# learn a linear predictor map by minimizing MSE incurred on corrupdated dataset D'
reg_cor = LinearRegression(fit_intercept=True)
reg_cor = reg_cor.fit(X_cor, y_cor)   
y_pred_cor = reg_cor.predict(X_cor)   

# Create a plot object 
fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 8))   # create a figure with two vertical subplots 

# Plot a subplot 1 with original data
axes.scatter(X, y, label='data')  # Plot data points
axes.plot(X, y_pred, color='green', label='linear predictor')  # Plot linear predictor

# For each data point, add line indicating prediction error
axes.plot((X[0], X[0]), (y[0], y_pred[0]), color='red', label='errors')  # Add label to legend
for i in range(len(X)-1):
    lineXdata = (X[i+1], X[i+1])  # Same X
    lineYdata = (y[i+1], y_pred[i+1])  # Different Y
    axes.plot(lineXdata, lineYdata, color='red')

# Set axes title, labels and legend
axes.set_title('linear regression on dataset $\mathcal{D}$')
axes.set_ylabel("$y$: Median apartment value (10000€)")
axes.set_ylim(10,40)  # set y-axis range to 0 till 100
axes.legend()


# # Plot a subplot 2 with corrupted data
# axes[1].scatter(X_cor, y_cor, label='data') 
# axes[1].set_ylim(10,40)  
# axes[1].plot(X, y_pred_cor, color='green')  

# # plot prediction errors
# for i in range(len(X)):
#     lineXdata = (X[i], X[i])  
#     lineYdata = (y_cor[i], y_pred_cor[i]) 
#     axes[1].plot(lineXdata, lineYdata, color='red')

# # set axes title, labels and legend
# axes[1].set_title('linear regression on perturbed dataset $\mathcal{D}{\'}$')
# axes[1].set_xlabel('$x_1$: Average number of rooms')
# axes[1].set_ylabel("$y$: Median apartment value (10000€)")

fig.tight_layout()
plt.show()
plt.show()

print("optimal weight w_opt by fitting to (training on) clean training data : ", reg.coef_[0])
print("optimal weight w_opt by fitting to (training on) corrupted training data : ", reg_cor.coef_[0])
