import numpy as np
import matplotlib.pyplot as plt


#------------------------------------------------------------
# Define the Huber loss
def HuberLoss(pred_error, epsilon):
    # pred_error - prediction error y-y_pred
    # epsilon - parameter epsilon 𝜀 
    pred_error = abs(pred_error)
    flag = (pred_error > epsilon)
    return (~flag) * (0.5 * pred_error ** 2) - (flag) * epsilon * (0.5 * epsilon - pred_error)

#------------------------------------------------------------
# Plot for several values of epsilon
fig = plt.figure(figsize=(10, 5)) # set figure size
ax = fig.add_subplot(111) # add 1 subplot

pred_error = np.linspace(-5, 5, 100) # create linear space from -5 to 5 with 100 steps

for epsilon in (1, 2, 10): # loop through values 1, 2, 10
    loss = HuberLoss(pred_error, epsilon)
    ax.plot(pred_error, loss, '-k') # plot x and y

    if epsilon > 10:
        s = r'\infty' # set s to infinity sign (string format)
    else:
        s = str(epsilon) # set s to string of number epsilon

    ax.text(pred_error[6], loss[6], '$\epsilon=%s$' % s,
            ha='center', va='center',
            bbox=dict(boxstyle='round', ec='k', fc='w')) # add test to each line

ax.plot(pred_error, np.square(pred_error),label="squared loss") # plot the sqared loss (blue line)

ax.set_xlabel(r'Error: $y - \hat{y}$') # set x labels
ax.set_ylabel(r'loss $\mathcal{L}(y,\hat{y})$') # set y label
ax.legend() # show legend in plot
plt.show() # show the plot
