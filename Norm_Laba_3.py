import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from random import uniform

true_a = 1
true_b = 2
true_c = 1
x_min = -5
x_max = 5
points = 50
x = np.linspace(x_min, x_max, points)
y = true_a * x ** 2 + true_b * x + true_c + np.array([uniform(-3, 3) for _ in range(points)])


def get_da(x, y, a, b, c):
    n = len(x)
    y_pred = a * x ** 2 + b * x + c
    return (2 / n) * np.sum(x ** 2 * (y_pred - y))


def get_db(x, y, a, b, c):
    n = len(x)
    y_pred = a * x ** 2 + b * x + c
    return (2 / n) * np.sum(x * (y_pred - y))


def get_dc(x, y, a, b, c):
    n = len(x)
    y_pred = a * x ** 2 + b * x + c
    return (2 / n) * np.sum(y_pred - y)


learning_rate = 0.001
epochs = 2000
initial_a = 0.0
initial_b = 0.0
initial_c = 0.0


def quadratic_fit(x, y, learning_rate, epochs, a0, b0, c0):
    a = a0
    b = b0
    c = c0
    a_history = [a]
    b_history = [b]
    c_history = [c]

    for _ in range(epochs):
        da = get_da(x, y, a, b, c)
        db = get_db(x, y, a, b, c)
        dc = get_dc(x, y, a, b, c)
        a -= learning_rate * da
        b -= learning_rate * db
        c -= learning_rate * dc
        a_history.append(a)
        b_history.append(b)
        c_history.append(c)

    return a_history, b_history, c_history


a_history, b_history, c_history = quadratic_fit(x, y, learning_rate, epochs, initial_a, initial_b, initial_c)
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.25)
scatter = ax.scatter(x, y, color='blue', label='Исходные данные')
regression_line, = ax.plot(x, a_history[0] * x ** 2 + b_history[0] * x + c_history[0], 'r-',
                           label='Квадратичная регрессия')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True)
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, 'Эпоха', 0, epochs, valinit=0, valstep=1)


def update(val):
    epoch = int(slider.val)
    current_a = a_history[epoch]
    current_b = b_history[epoch]
    current_c = c_history[epoch]
    regression_line.set_ydata(current_a * x ** 2 + current_b * x + current_c)
    y_pred = current_a * x ** 2 + current_b * x + current_c
    mse = np.mean((y_pred - y) ** 2)
    ax.set_title(
        f'Эпоха {epoch})\n'
        f'a = {current_a:.3f} '
        f'b = {current_b:.3f} '
        f'c = {current_c:.3f}'
    )

    fig.canvas.draw_idle()


slider.on_changed(update)
update(0)
plt.show()
