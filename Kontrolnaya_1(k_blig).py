import random
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

xMin1, xMax1, yMin1, yMax1, xMin2, xMax2, yMin2, yMax2, p = 0, 10, 0, 10, 5, 15, 5, 15, 0.8
model = KNeighborsClassifier(n_neighbors=3)
x = []
y = []
pointsCount1 = []
pointsCount2 = []
for i in range(50):
    pointsCount1.append([random.uniform(xMin1, xMax1), random.uniform(yMin1, yMax1)])
    y.append(0)
    pointsCount2.append([random.uniform(xMin2, xMax2), random.uniform(yMin2, yMax2)])
    y.append(1)
for i, j in zip(pointsCount1, pointsCount2):
    x.append(i)
    x.append(j)


def fit(a, b, c):
    model.fit(a, b)
    d = model.predict(c)
    return d


x_train, y_train, x_test, y_test = x[:int(100 * p)], y[:int(100 * p)], x[int(100 * p):], y[int(100 * p):]
y_predict = fit(x_train, y_train, x_test)
print(y_test)
print(y_predict)
accuracy = accuracy_score(y_test, y_predict)
print(accuracy)

xtrx = []
xtry = []
xtx = []
xty = []
n = 0
for i in x_train:
    xtrx.append(i[0])
    xtry.append(i[1])
for i in x_test:
    xtx.append(i[0])
    xty.append(i[1])
for i in y_train:
    if i == 0:
        plt.scatter(xtrx[n], xtry[n], c='blue', marker='o')
    else:
        plt.scatter(xtrx[n], xtry[n], c='blue', marker='x')
    n += 1
n = 0
for i, j in zip(y_predict, y_test):
    if i != j:
        if j == 0:
            plt.scatter(xtx[n], xty[n], c='red', marker='o')
        else:
            plt.scatter(xtx[n], xty[n], c='red', marker='x')
    n += 1
n = 0
for i, j in zip(y_predict, y_test):
    if i == j:
        if i == 0:
            plt.scatter(xtx[n], xty[n], c='green', marker='o')
        else:
            plt.scatter(xtx[n], xty[n], c='green', marker='x')
    n += 1
plt.grid()
plt.show()
