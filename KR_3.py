import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
plt.imshow(X_train[0].reshape(8, 8))
plt.show()
plt.imshow(X_test[0].reshape(8, 8))
plt.show()

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_knn)
precision = precision_score(y_test, y_pred_knn, average='weighted')
recall = recall_score(y_test, y_pred_knn, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred_knn)
print("KNN:")
print(f"accuracy: {round(accuracy)}")
print(f"precision: {round(precision)}")
print(f"recall: {round(recall)}")
print(conf_matrix)


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_rf)
precision = precision_score(y_test, y_pred_rf, average='weighted')
recall = recall_score(y_test, y_pred_rf, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred_rf)
print("Random Forest:")
print(f"accuracy: {round(accuracy)}")
print(f"precision: {round(precision)}")
print(f"recall: {round(recall)}")
print("Матрица ошибок:")
print(conf_matrix)


def vivod(y_true, y_pred, metod):
    print(f"2 верно классифицированных примера {metod}")
    for i in np.random.choice(np.where(y_true == y_pred)[0], 2, replace=False):
        plt.imshow(X_test[i].reshape(8, 8))
        plt.show()
    print(f"2 неверно классифицированных примера {metod}")
    for i in np.random.choice(np.where(y_true != y_pred)[0], 2, replace=False):
        plt.imshow(X_test[i].reshape(8, 8))
        plt.show()


vivod(y_test, y_pred_knn, "KNN")
vivod(y_test, y_pred_rf, "Random Forest")

both_wrong = np.where((y_test != y_pred_knn) & (y_test != y_pred_rf))[0]
if len(both_wrong) > 0:
    print("Неверно классифицированные:")
    for i in both_wrong[:min(2, len(both_wrong))]:
        plt.imshow(X_test[i].reshape(8, 8))
        plt.show()
