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
from sklearn import metrics
from sklearn import datasets
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline


features_path = "image_data.csv"
labels_path = "image_labels.csv"

def load_data(binary_labels=False):
    X = pd.read_csv(features_path, header=None).to_numpy()
    y = pd.read_csv(labels_path, header=None).to_numpy().reshape(-1,)
    
    # select first 5 features
    X = X[:,:5]
    
    # convert labels to (new) binary labels
    # label for class 0 is y=1 and label for class 1 and class 2 is y=0.
    if binary_labels:
        y = (y == 0).astype(int)
        
    return X, y

#X, y = load_data()

# # Print information of dataset
# print(f"Shape of feature matrix: {X.shape} \nShape of label vector: {y.shape}")
# print(f"Number of samples from Class 0: {sum(y == 0)}")
# print(f"Number of samples from Class 1: {sum(y == 1)}")
# print(f"Number of samples from Class 2: {sum(y == 2)}")


# Load data
#X, y = load_data(binary_labels=True)

# idx_1 = np.where(y == 1) # Indices of no pedestrian crossings images
# idx_2 = np.where(y == 0) # Indices of one or more crossings images

# # Plot scatterplot of dataset with different markings for classes
# fig, axes = plt.subplots(figsize=(10, 6))
# axes.scatter(X[idx_1, 0], X[idx_1, 1], c='green', marker ='o', label='y =1; No crossings')
# axes.scatter(X[idx_2, 0], X[idx_2, 1], c='brown', marker ='x', label='y =0; One or more crossings ')

# # Set axis labels and legend
# axes.legend(loc='upper left', fontsize=12)
# axes.set_xlabel('feature $x_1$', fontsize=16)
# axes.set_ylabel('feature $x_2$', fontsize=16)
# plt.show()


# Set random seed for reproducibility
# np.random.seed(0)

# # Load the features and labels
X, y = load_data(binary_labels=True)

# # Split dataset into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=0)

# ### BEGIN STUDENT TASK ###

# scaler = StandardScaler()
# log_reg = LogisticRegression(random_state=0, C=1e6)
# pipe = make_pipeline(scaler, log_reg)
# pipe.fit(X_train, y_train)

# acc_train = pipe.score(X_train, y_train)
# acc_test = pipe.score(X_test, y_test)


# # Print training and validation errors
# print(f"Training accuracy: {acc_train}")
# print(f"Test accuracy: {acc_test}")


# np.random.seed(0)

# # Split dataset into train-val and test sets
# X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y, random_state=42)

# # Candidates for the inverse regularization strength
# C_candidates = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

# ### STUDENT TASK ###
# scaler = StandardScaler()
# log_reg = LogisticRegression(random_state=0, multi_class='ovr')

# pipe = pipe = Pipeline(steps=[("scaler", scaler), ("log_reg", log_reg)])

# param_grid = {
#     "log_reg__C": C_candidates,
# }

# search = GridSearchCV(pipe, param_grid, refit=True, cv=5, return_train_score=True)
# search.fit(X_trainval, y_trainval)

# acc_val = search.cv_results_['mean_test_score']
# acc_train = search.cv_results_['mean_train_score']

# best_model = search.best_estimator_

# # Print training and validation errors
# print(f"Training accuracy: {acc_train}")
# print(f"Validation accuracy: {acc_val}")
# print(f"Best model: {best_model}")


# assert len(acc_train) == len(C_candidates), "acc_train is of the wrong size!"
# assert len(acc_val) == len(C_candidates), "acc_val is of the wrong size!"
# assert best_model.get_params()['log_reg__C'] == 1, "The optimal parameter value is wrong!"
# print('Sanity check tests passed!')

def plot_decision_boundary(clf, X, Y, cmap='Paired_r'):
    
    """
    Function with which to plot decision boundary
    """
    
    # step size
    h = 0.02
    # min-max values of features x1 and x2
    x1_min, x1_max = X[:,0].min() - 10*h, X[:,0].max() + 10*h
    x2_min, x2_max = X[:,1].min() - 10*h, X[:,1].max() + 10*h
    # x1, x2 values for plotting 
    x1 = np.arange(x1_min, x1_max, h)
    x2 = np.arange(x2_min, x2_max, h)
    # create grid of x1,x2 values (all possible combinations of x1 and x2 values)
    x1x1, x2x2 = np.meshgrid(x1, x2)
    # get predictions for each x1,x2 pair
    Z = clf.predict(np.c_[x1x1.ravel(), x2x2.ravel()])
    Z = Z.reshape(x1x1.shape)
    
    idx_1 = np.where(Y == 1)[0] # index of each class 0 iamge.
    idx_2 = np.where(Y == 0)[0] # index of each not class 0 image
    
    plt.figure(figsize=(10,6))
    plt.contourf(x1x1, x2x2, Z, cmap=cmap, alpha=0.25)
    plt.contour(x1x1, x2x2, Z, colors='k', linewidths=0.5)
    plt.scatter(X[idx_1, 0], X[idx_1, 1], marker='x', label='class 0')
    plt.scatter(X[idx_2, 0], X[idx_2, 1], marker='o', label='class 1', edgecolors='k')
    plt.xlabel(r'Feature 1')
    plt.ylabel(r'Feature 2')


# Set random seed
np.random.seed(0)

# Load data and select only the first two features
X, y = load_data(binary_labels=True)
# X = X[:,:2]

# clf = DecisionTreeClassifier(criterion='entropy')   # define object "clf" which represents a decision tree
# clf.fit(X, y)                    # learn a decision tree that fits well the labeled images  
# y_pred = clf.predict(X)          # compute the predicted labels for the images

# # Calculate the accuracy score of the predictions
# accuracy = clf.score(X, y)
# print(f"Accuracy: {round(100*accuracy, 2)}%")

# # Plot decision boundary
# plot_decision_boundary(clf, X, y)
# plt.show()


# Set random seed
np.random.seed(0)

# Load data to feature matrix X and label vector y 
X, y = load_data(binary_labels=True)

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=0)

### STUDENT TASK ###
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)  
y_pred = clf.predict(X_train)     

# accuracy = 
acc_train = clf.score(X_train, y_train)
acc_test = clf.score(X_test, y_test)


# Print training and validation error
print(f"Training error: {acc_train}")
print(f"Test error: {acc_test}")

assert acc_train > 0.99, "Training accuracy is too low."
assert acc_test > 0.9, "Test accuracy is too low."
assert acc_test < 1, "Test accuracy is too high."
print('Sanity check tests passed!')

plot_decision_boundary(clf, X_test, y_test)
plt.show()