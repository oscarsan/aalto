from sklearn import linear_model
from sklearn.linear_model import HuberRegressor
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

def load_corrupted_data():
    X, y = load_housing_data(n=1)
    
    # perturb the label of the data point with lowest x_1
    y[np.argmin(X)] = 80
    
    return X, y


X, y = load_housing_data(n=1)
X_cor, y_cor = load_corrupted_data()   # read in 20 data points with single feature x_1 and label y 

### STUDENT TASK ###
# Huber regression model on the original dataset
reg = HuberRegressor(fit_intercept=True) 
reg.fit(X, y)
w_opt = reg.coef_
w_intercept = reg.intercept_
y_pred = reg.predict(X)


# Huber regression model on the corrupted dataset
reg_cor = HuberRegressor(fit_intercept=True) 
reg_cor.fit(X_cor, y_cor)
w_opt_cor = reg_cor.coef_
w_intercept_cor = reg_cor.intercept_
y_pred_cor = reg_cor.predict(X_cor)


print("optimal weight w_opt by fitting on clean data : ", reg.coef_[0])
print("optimal weight w_opt by fitting on perturbed data : ", reg_cor.coef_[0])
