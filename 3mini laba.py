# f(x) = a + b * (x ** (3 / 2)) + c * x ** 3
import matplotlib.pyplot as plt
import numpy as np

a = int(input())
b = int(input())
c = int(input())
x = np.linspace(0, 2 * np.pi, 200)
y = a + b * (x ** (3 / 2)) + c * x ** 3
plt.plot(x, y)
plt.show()
