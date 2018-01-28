import numpy as np

def generate_dataset(coeffs, n, std_dev):
  # We calculate the number of predictors, and create a coefficient matrix
  # With `p` rows and 1 column, for matrix multiplication
  p = len(coeffs)
  coeff_mat = np.array(coeffs).reshape(p, 1)
  # Similar as before, but with `n` rows and `p` columns this time
  x = np.random.random_sample((n, p))* 100
  e = np.random.randn(n) * std_dev
  # Since x is a n*p matrix, and coefficients is a p*1 matrix
  # we can use matrix multiplication to get the value of y for each
  # set of values x1, x2 .. xp
  # We need to transpose it to get a 1*n array from a n*1 matrix to use in the regression model
  y = np.matmul(x, coeff_mat).transpose() + e
  return x, y

def generate_dataset_simple(beta, n, std_dev):
  # Generate x as an array of `n` samples which can take a value between 0 and 100
  x = np.zeros(np.random() * 100)
  # Generate the random error of n samples, with a random value from a normal distribution, with a standard
  # deviation provided in the function argument
  e = np.random.randn(n) * std_dev
  # Calculate `y` according to the equation discussed
  # We need to reshape x to be a 2-d matrix with n rows and 1 column
  # This is so that it can take a generalized form that can be expanded
  # to multiple predictors for the `LinearRegression` model
  return x.reshape(n, 1), y