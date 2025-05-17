import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import random


# f(x) = x*sin(x) + log(x+1)
def f(x):
    return x * np.sin(x) + np.log(x + 1)


is_linear = False
x = np.linspace(1, 10, 100).reshape(-1, 1)
noise = np.array([random.uniform(-0.5, 0.5) for _ in range(100)]).reshape(-1, 1)
y = f(x) + noise
models = [
    ("Линейная регрессия", LinearRegression()),
    ("Метод опорных векторов (SVR)", SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)),
    ("Случайный лес", RandomForestRegressor(n_estimators=100))
]
plt.figure(figsize=(15, 10))

for i, (name, model) in enumerate(models):
    model.fit(x, y.ravel())
    y_pred = model.predict(x)
    mse = mean_squared_error(y, y_pred)
    plt.subplot(3, 1, i + 1)
    plt.scatter(x, y, color='blue', label='Зашумленные данные', alpha=0.6)
    x_plot = np.linspace(1, 10, 300)
    plt.plot(x_plot, f(x_plot), 'g-', linewidth=2, label='Исходная функция')
    plt.plot(x, y_pred, 'r-', linewidth=2, label=f'Предсказание ({name})')
    plt.title(name)
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
