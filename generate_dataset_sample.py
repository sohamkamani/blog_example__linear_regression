from mpl_toolkits.mplot3d import Axes3D
from generate_dataset import generate_dataset as gd
from matplotlib import pyplot as plt

predictor_coeffs =[1, 1]
std_dev = 10
n = 100

X, Y = gd(predictor_coeffs, n, std_dev)


f, [[p1, p2], [p3, p4]] = plt.subplots(2,2)

p1.scatter(X[:,0], Y)
p1.set_title("x1")

p2.scatter(X[:,1], Y)
p2.set_title("x2")

p3 = f.add_subplot(223, projection='3d')
p3.scatter(X[:,0], X[:,1], Y)
p3.set_title("x1, x2, y")

plt.show()