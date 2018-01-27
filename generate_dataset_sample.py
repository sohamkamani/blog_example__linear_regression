from generate_dataset import generate_dataset as gd
from matplotlib import pyplot as plt

predictor_coeffs =[10, 1, 0, 6]
std_dev = 10
n = 100

X, Y = gd(predictor_coeffs, n, std_dev)

print("X: ", X[:10, 2], "Y: ", Y[:10])

f, [[p1, p2], [p3, p4]] = plt.subplots(2,2)

p1.scatter(X[:,0], Y)
p1.set_title("BB : " + str(predictor_coeffs[0]))

p2.scatter(X[:,1], Y)
p2.set_title("BB : " + str(predictor_coeffs[1]))

p3.scatter(X[:,2], Y)
p3.set_title("BB : " + str(predictor_coeffs[2]))

p4.scatter(X[:,3], Y)
p4.set_title("BB : " + str(predictor_coeffs[3]))

plt.show()