from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("C:\\Users\\James Calap\\Desktop\\test.csv")
df = df.dropna()

reg = linear_model.LinearRegression()
reg.fit(df[['year']], df.capita)

pre = reg.predict([[2020]])
print(pre)
plt.scatter(df.year, df.capita)
plt.plot(df.year, reg.predict(df[['year']]), color='red')
plt.show()