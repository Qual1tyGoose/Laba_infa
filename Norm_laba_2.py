import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def update(val):
    epoch = int(slider.val)
    ax.clear()
    centroids, labels = history[epoch]
    colors = ['red', 'green', 'blue']
    for i in range(k):
        cluster_points = data[labels == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                   c=colors[i])
    ax.scatter(centroids[:, 0], centroids[:, 1],
               c='black', marker='x', s=100, linewidths=2)

    ax.set_title(f'Иттерации {epoch}')
    ax.legend()


centers = [(2, 2), (5, 5), (8, 2)]
radii = [1, 1, 1]
num_points = 20
xx, yy = [], []
index = 0
for i in centers:
    for _ in range(num_points):
        xx.append(i[0] + np.random.uniform(-1) * radii[index])
        yy.append(i[1] + np.random.uniform(-1) * radii[index])
    index += 1
x, y = np.array(xx), np.array(yy)
data = np.column_stack((x, y))
k = 3


def k_means():
    centr = data[np.random.choice(len(data), k, replace=False)]
    label = []
    for _ in range(100):
        dist = np.sqrt(((data - centr[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(dist, axis=0)
        label.append((centr.copy(), labels.copy()))
        ncentr = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        centr = ncentr

    return label


history = k_means()
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.2)
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Иттерации', 0, len(history) - 1, valinit=0, valstep=1)
slider.on_changed(update)
update(0)
plt.show()
