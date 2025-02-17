import matplotlib.pyplot as plt
import numpy as np

file = open('dlya_3mini_laba_2.txt', 'r')
xy = []
for i in file.readlines():
    xy.append(i.split())
x = list(map(int, xy[0]))
y = list(map(int, xy[-1]))
file.close()
sizes = np.random.uniform(15, 80, len(x))
colors = np.random.uniform(15, 80, len(x))
plt.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=100)
plt.show()
