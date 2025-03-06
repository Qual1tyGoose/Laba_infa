import random
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

xMin1 = 0
xMax1 = 10
yMin1 = 0
yMax1 = 10
xMin2 = 5
xMax2 = 15
yMin2 = 5
yMax2 = 15
p = 0.8
model = KNeighborsClassifier(n_neighbors=3)
random.seed(110)
pointsCount1 = []
pointsCount2 = []
x_train, y_train, x_test, y_test = 0, 0, 0, 0
y = []
for i in range(50):
    pointsCount1.append([random.uniform(xMin1, xMax1), random.uniform(yMin1, yMax1)])
    pointsCount2.append([random.uniform(xMin2, xMax2), random.uniform(yMin2, yMax2)])
    y.append(random.randint(0, 1))
    y.append(random.randint(0, 1))
x = []
for i, j in zip(pointsCount1, pointsCount2):
    x.append(i)
    x.append(j)


def train_test_split(xx, yy):
    return xx[:int(100 * p)], yy[:int(100 * p)], xx[int(100 * p):], yy[int(100 * p):]


def fitt(xx_train, yy_train, xx_test):
    model.fit(xx_train, yy_train)
    return model.predict(xx_test)


def computeAccuracy(yy_test, yy_predict):
    return accuracy_score(yy_test, yy_predict)


x_train, y_train, x_test, y_test = train_test_split(x, y)
y_predict = fitt(x_train, y_train, x_test)
accuracy = computeAccuracy(y_test, y_predict)
plt.figure(figsize=(10, 6))

x_train1 = []
x_train2 = []
for i in x_train:
    x_train1.append(i[0])
    x_train2.append(i[1])
x_test1 = []
x_test2 = []
for i in x_test:
    x_test1.append(i[0])
    x_test2.append(i[1])
x_train1o = []
x_train2o = []
x_train1x = []
x_train2x = []
index = 0
for i in y_train:
    if i == 0:
        x_train1o.append(x_train1[index])
        x_train2o.append(x_train2[index])
    else:
        x_train1x.append(x_train1[index])
        x_train2x.append(x_train2[index])
    index += 1
plt.scatter(x_train1o, x_train2o, marker='o', c='blue')
plt.scatter(x_train1x, x_train2x, marker='x', c='blue')
x_test1ok = []
x_test2ok = []
x_test1xk = []
x_test2xk = []
x_test1oz = []
x_test2oz = []
x_test1xz = []
x_test2xz = []
index = 0
for i, j in zip(y_predict, y_test):
    if i != j:
        if i == 0:
            x_test1ok.append(x_test1[index])
            x_test2ok.append(x_test2[index])
        else:
            x_test1xk.append(x_test1[index])
            x_test2xk.append(x_test2[index])
    else:
        if i == 0:
            x_test1oz.append(x_test1[index])
            x_test2oz.append(x_test2[index])
        else:
            x_test1xz.append(x_test1[index])
            x_test2xz.append(x_test2[index])
    index += 1
plt.scatter(x_test1ok, x_test2ok, marker='o', c='red')
plt.scatter(x_test1xk, x_test2xk, marker='x', c='red')
plt.scatter(x_test1xz, x_test2xz, marker='x', c='green')

# Настройка графика
plt.title('Классификация точек с помощью k-NN')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.axhline(0, color='grey', lw=0.5)
plt.axvline(0, color='grey', lw=0.5)
plt.grid()
plt.show()
