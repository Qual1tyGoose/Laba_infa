import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv(r"датасет-1.csv", sep=',')
print(df)
print(df.dtypes)
df['price'] = df['price'].astype(float)
plt.scatter(df.area, df.price, color='red')
plt.xlabel('площадь(кв.м.)')
plt.show()
reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)
print(reg.predict([[38]]))
print(reg.predict(df[['area']]))
print(reg.coef_)
print(reg.intercept_)
plt.scatter(df.area, df.price, color='red')
plt.xlabel('площадь(кв.м.)')
plt.ylabel('стоимость(млн.руб)')
plt.plot(df.area, reg.predict(df[['area']]))
plt.show()
pred = pd.read_csv(r"prediction_price.csv", sep=';')
print(pred)
pred['area'] = pred['area'].astype(float)
p = reg.predict(pred)
pred['predicted prices'] = p
print(pred)
pred.to_excel('new.xlsx', index=False)
