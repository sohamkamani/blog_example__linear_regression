import numpy as np

def generate_dataset(coeffs, n, std_dev, range=(0,100)):
  p = len(coeffs)
  X = np.zeros((n, p))
  X = [np.array([np.random.random_sample()*100 for _ in row]) for row in X]
  Y = [np.dot(row, coeffs) + np.random.randn() * std_dev for row in X]
  return np.array(X), np.array(Y)
