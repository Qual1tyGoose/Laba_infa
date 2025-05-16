import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from random import uniform

a = 2
b = 0.3
c = 1

x_min = 0
x_max = 6
points = 50
noise_scale = 3

x = np.linspace(x_min, x_max, points)
y = a * np.exp(b * x) + c + np.array([uniform(-noise_scale, noise_scale) for _ in range(points)])


def get_da(x, y, a, b, c):
    exp_bx = np.exp(b * x)
    y_pred = a * exp_bx + c
    return (2 / len(x)) * np.sum(exp_bx * (y_pred - y))


def get_db(x, y, a, b, c):
    exp_bx = np.exp(b * x)
    y_pred = a * exp_bx + c
    return (2 / len(x)) * np.sum(a * x * exp_bx * (y_pred - y))


def get_dc(x, y, a, b, c):
    exp_bx = np.exp(b * x)
    y_pred = a * exp_bx + c
    return (2 / len(x)) * np.sum(y_pred - y)


learning_rate = 0.001
epochs = 1000
initial_a = 1.0
initial_b = 0.1
initial_c = 0.5


def exponential_fit(x, y, learning_rate, epochs, a0, b0, c0):
    a = a0
    b = b0
    c = c0
    history = {'a': [a], 'b': [b], 'c': [c]}
    for _ in range(epochs):
        da = get_da(x, y, a, b, c)
        db = get_db(x, y, a, b, c)
        dc = get_dc(x, y, a, b, c)
        a -= learning_rate * da
        b -= learning_rate * db
        c -= learning_rate * dc
        history['a'].append(a)
        history['b'].append(b)
        history['c'].append(c)

    return history


history = exponential_fit(x, y, learning_rate, epochs, initial_a, initial_b, initial_c)
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.25)
scatter = ax.scatter(x, y, color='blue', label='Исходные точки')
regression_line, = ax.plot(x, history['a'][0] * np.exp(history['b'][0] * x) + history['c'][0],
                           'r-', label='Показательная регрессия')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True)
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, 'Эпоха', 0, epochs, valinit=0, valstep=1)


def update(val):
    epoch = int(slider.val)
    current_a = history['a'][epoch]
    current_b = history['b'][epoch]
    current_c = history['c'][epoch]
    regression_line.set_ydata(current_a * np.exp(current_b * x) + current_c)
    ax.set_title(
        f'Показательная регрессия (Эпоха: {epoch})\n'
        f'a = {current_a:.4f}, '
        f'b = {current_b:.4f}, '
        f'c = {current_c:.4f}'
    )
    fig.canvas.draw_idle()


slider.on_changed(update)
update(0)
plt.show()
