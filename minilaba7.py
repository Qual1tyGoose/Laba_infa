import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

n_samples = 500
seed = 30


def generate_circles():
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed)
    return noisy_circles


def generate_moons():
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
    return noisy_moons


def generate_blobs():
    cluster_std = [1.0, 0.5]
    varied = datasets.make_blobs(n_samples=n_samples, cluster_std=cluster_std, random_state=seed, centers=2)
    return varied


def generate_varied_density():
    random_state = 170
    x, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state, centers=2)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    x_aniso = np.dot(x, transformation)
    aniso = (x_aniso, y)
    return aniso


def generate_s_shape():
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed, centers=2)
    return blobs


datasets = [
    ("Две окружности", generate_circles()),
    ("Две параболы", generate_moons()),
    ("Хаотичное распределение", generate_blobs()),
    ("Разная плотность", generate_varied_density()),
    ("S-образное распределение", generate_s_shape())
]

classifiers = [
    ("KNN", KNeighborsClassifier(n_neighbors=3)),
    ("Логистическая регрессия", LogisticRegression(max_iter=200)),
    ("Наивный Байес", GaussianNB())
]

fig, axes = plt.subplots(len(datasets), len(classifiers), figsize=(15, 20))
for row_idx, (data_name, (X, y)) in enumerate(datasets):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100),
                         np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 100))
    for col_idx, (clf_name, model) in enumerate(classifiers):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        ax = axes[row_idx, col_idx]
        Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', alpha=0.6)
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='bwr', marker='s')
        wrong = y_pred != y_test
        ax.scatter(X_test[wrong, 0], X_test[wrong, 1], edgecolors='black', s=80)
        ax.set_title(f"{data_name}\n{clf_name} | Точность: {accuracy:.2f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.show()
