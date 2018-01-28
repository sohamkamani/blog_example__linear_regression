print(__doc__)
# FIlter stupic warnings
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

import matplotlib.pyplot as plt
import numpy as np
from generate_dataset import generate_dataset as gd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

predictor_coeffs =[10, 1, 0, 6]
std_dev = 10
n = 200

X, Y = gd(predictor_coeffs, n, std_dev)

x_trn = X[:-20]
y_trn = Y[:-20]

x_test = X[-20:]
y_test = Y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_trn, y_trn)

# Make predictions using the testing set
y_pred = regr.predict(x_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

# Plot outputs
# plt.scatter(diabetes_x_test, diabetes_y_test,  color='black')
# plt.plot(diabetes_x_test, diabetes_y_pred, color='blue', linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()