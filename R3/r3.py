# Import basic libraries needed in this round
import numpy as np 
import pandas as pd  
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split  
#pdb.set_trace()
import pdb 
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import PolynomialFeatures


def load_housing_data(n=10):
    # Load dataframe from csv
    df = pd.read_csv("../helsinki_apartment_prices.csv", index_col=0)  
    
    # Extract feature matrix and label vector from dataframe
    X = df.iloc[:,:n].to_numpy()
    y = df.iloc[:,-1].to_numpy().reshape(-1)
    
    return X, y
    
# Load the housing data
X, y = load_housing_data()
#print(X.shape, y.shape)




# n = 10                        # maximum number of features used 

# X,y = load_housing_data(n=n)  # read in 20 data points using n features 
# linreg_error = np.zeros(n)    # vector for storing the training errors

# for i in range(n): 
#     reg = LinearRegression(fit_intercept=True)    # create an object for linear predictors
#     reg = reg.fit(X[:,:(i+1)], y)    # find best linear predictor (minimize training error)
#     pred = reg.predict(X[:,:(i+1)])    # compute predictions of best predictors 
#     linreg_error[i] = mean_squared_error(y, pred)    # compute training error 

# plot_x = np.linspace(1, n, n, endpoint=True)    # plot_x contains grid points for x-axis (1,...,n)

# # Plot training error E(r) as a function of feature number r
# plt.rc('legend', fontsize=14)    # Set font size for legends
# plt.rc('axes', labelsize=14)     # Set font size for axis labels
# plt.figure(figsize=(8,4))        # Set figure size
# plt.plot(plot_x, linreg_error, label='$E(r)$', color='red')
# plt.xlabel('# of features $r$')
# plt.ylabel('training error $E(r)$')
# plt.title('training error vs number of features', fontsize=16)
# plt.legend()
# plt.show()
#X, y = load_housing_data()

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2)

# assert len(X_train) == 16, "The 'X_train' vector has the wrong length"
# assert len(y_train) == 16, "The 'y_train' vector has the wrong length"
# assert len(X_val) == 4,   "The 'X_val' vector has the wrong length"
# assert len(y_val) == 4, "The 'y_val' vector has the wrong length"
# print('Sanity checks passed!')

# def get_train_val_errors(X_train, X_val, y_train, y_val, n_features):  
#     err_train = np.zeros(n_features)  # Array for storing training errors
#     err_val = np.zeros(n_features)    # Array for storing validation errors
    
#     for i in range(n_features):    # Loop over the number of features n features
#         reg = LinearRegression(fit_intercept=True)    
#         reg = reg.fit(X_train[:,:(i+1)], y_train)
#         pred = reg.predict(X_train[:,:(i+1)])
#         err_train[i] = mean_squared_error(y_train, pred)    # compute training error
#         pred_val = reg.predict(X_val[:,:(i+1)]) 
#         err_val[i] = mean_squared_error(y_val, pred_val)    # compute training error 


#     return err_train, err_val

# n = 10

# # Calculate training and validation errors using ´get_train_val_errors´
# err_train,err_val = get_train_val_errors(X_train, X_val, y_train, y_val, n)



# # Perform some sanity checks on the results
# assert err_train.shape == (n,), "numpy array err_train has wrong shape"
# assert err_val.shape == (n,), "numpy array err_val has wrong shape"
# print('Sanity checks passed!')


# # Plot the training and validation errors for the different number of features r
# plt.figure(figsize=(8,4))
# plt.plot(range(1, n + 1), err_train, color='black', label=r'$E_{\rm train}(r)$', marker='o')  # Plot train error
# plt.plot(range(1, n + 1), err_val, color='red', label=r'$E_{\rm val}(r)$', marker='x')  # Plot validation error

# plt.title('Training and validation error for different number of features', fontsize=16)    # Set title
# plt.ylabel('Empirical error')    # Set label for y-axis
# plt.xlabel('r features')    # Set label for x-axis
# plt.xticks(range(1, n + 1))  # Set the tick labels on the x-axis to be 1,...,n
# plt.legend()
# plt.show()


# n = 10 # max number of features
# X, y = load_housing_data(n=n)  # read in 20 data points with n features 

# err_train = np.zeros(n)  # Array to store training errors
# err_val = np.zeros(n)  # Array to store validation errors

# K = 5  # Number of splits
# kf = KFold(n_splits=K, shuffle=False)    # Create a KFold object with 'K' splits

# iteration = 0
# for train_indices, val_indices in kf.split(X):
#     iteration += 1
#     X_train = X[train_indices,:]    # Get the training set    
#     X_val = X[val_indices,:]    # Get the validation set
#     print('Iteration {}:'.format(iteration))
#     print('Indices for validation set:', val_indices)
#     print('Indices for training set:', train_indices)
#     print('X_val shape: {}, X_train shape: {} \n'.format(X_val.shape, X_train.shape))

#pdb.set_trace()


# n = 10 # max number of features
# X, y = load_housing_data(n=n)  # read in 20 data points with n features 

# err_train = np.zeros(n)  # Array to store training errors
# err_val = np.zeros(n)  # Array to store validation errors

# K = 5  # Number of splits
# kf = KFold(n_splits=K, shuffle=False)  

# iteration = 0   
# for i in range(n):    
#     err_train_splits = []  
#     err_val_splits = []
#     for train_indices, val_indices in kf.split(X):
#         iteration += 1
#         X_train = X[train_indices,:]
#         y_train = y[train_indices]
#         X_val = X[val_indices,:]    
#         y_val = y[val_indices] 
#         reg = Lasso(alpha=0.1, fit_intercept=True)    
#         reg = reg.fit(X_train[:,:(i+1)], y_train)
#         pred = reg.predict(X_train[:,:(i+1)])
#         err_train_splits.append(mean_squared_error(y_train, pred))   
#         pred_val = reg.predict(X_val[:,:(i+1)]) 
#         err_val_splits.append(mean_squared_error(y_val, pred_val))
#     err_train[i] = np.mean(err_train_splits)
#     err_val[i] = np.mean(err_val_splits)



# print('Training errors for each K:')
# print(err_train, '\n')
# print('Validation error for each K:')
# print(err_val, '\n')



# n = 10
# X, y = load_housing_data(n)
# # 80% training and 20% val
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2)  

# alpha = 10    

# ridge = Ridge(alpha=alpha, fit_intercept=True)    
# ridge.fit(X_train, y_train)                       
# y_pred = ridge.predict(X_train)                   
# w_opt = ridge.coef_                               
# err_train = mean_squared_error(y_pred, y_train)   

#pdb.set_trace()

# Print optimal weights and training error
# # print('Optimal weights: \n', w_opt)
# # print('Training error: \n', err_train)
# n=10
# X, y = load_housing_data(n)
# poly = PolynomialFeatures(2, include_bias=False)
# X = poly.fit_transform(X)


# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2)    

# def fit_lasso(X, y, alpha_val):
#     lasso = Lasso(alpha=alpha_val, fit_intercept=True)    
#     lasso.fit(X_train, y_train)                       
#     y_pred = lasso.predict(X_train)                   
#     w_opt = lasso.coef_                             
#     error  = mean_squared_error(y_pred, y_train)   
#     return w_opt, error, y_pred

# alpha_val = 0.47

# # Fit Lasso and calculate optimal weights and training error 
# w_opt, training_error, y_pred = fit_lasso(X_train, y_train, alpha_val)
# pdb.set_trace()


#  # Print optimal weights and the corresponding training error
# print('Optimal weights: \n', w_opt)
# print('Training error: \n', training_error)

# # Create a plot object 
# fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 8))   # create a figure with two vertical subplots 

# # Plot a subplot 1 with original data
# axes.scatter(X_train[:,0], y_train, label='data')  # Plot data points

# axes.scatter(X_train[:,0], y_pred, color='green', label='linear predictor')  # Plot linear predictor

# # For each data point, add line indicating prediction error
# #axes.plot((X[0], X[0]), (y[0], y_pred[0]), color='red', label='errors')  # Add label to legend
# # for i in range(len(X_train)-1):
# #     lineXdata = (X_train[i+1], X_train[i+1])  # Same X
# #     lineYdata = (y_train[i+1], y_pred[i+1])  # Different Y
# #     axes.plot(lineXdata, lineYdata, color='red')


# fig.tight_layout()
# plt.show()
n=10
X, y = load_housing_data(n)    
poly = PolynomialFeatures(2, include_bias=False)
X = poly.fit_transform(X)

#alpha_values = [0.001, 0.01, 0.1, 1, 10, 100]
alpha_values = np.logspace(-4, 4, 100, endpoint=True)
#alpha_values = [0.1]
### STUDENT TASK ###
# 1. Create a dictionary 
params = {'alpha': alpha_values}

# 2. Create a Lasso object
lasso = Lasso(fit_intercept=True)   

# 3. Create a GridSearchCV object
ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cv = GridSearchCV(lasso, param_grid=params, scoring='neg_mean_squared_error', cv=ss, return_train_score=True)

# 4. fit a GridSearchCV object to data (X,y)
cv.fit(X,y)

# 5. retrieve training and validation errors from fitted GridSearchCV object 
err_val = cv.cv_results_['mean_test_score']*-1
err_train = cv.cv_results_['mean_train_score']*-1

print("this is the err_train ")
print(err_train, '\n')
print("this is the err_val ")
print(err_val, '\n')

assert err_train[0] > 0 and err_val[0] > 0, "Errors are negative!"
assert err_train.shape == (len(alpha_values),), "'err_train' has wrong shape"
assert err_val.shape == (len(alpha_values),), "'err_val' has wrong shape"
print('Sanity check tests passed!')

pdb.set_trace()


# Plot the training and validation errors
plt.figure(figsize=(8,4))    # Set figure size
plt.plot(alpha_values, err_train, marker='o', color='black', label='training error')    # Plot training errors
plt.plot(alpha_values, err_val, marker='o', color='red', label='validation error')    # Plot validation errors
plt.xscale('log')    # Set x-axis to logarithmic scale
plt.xlabel(r'$\alpha$')    # Set label of x-axis
plt.ylabel(r'$E(\alpha)$')    # Set label of y-axis
plt.title(r'Errors with respect to $\alpha$', fontsize=16)    # Set title
plt.legend()    # Show legend
plt.show()

