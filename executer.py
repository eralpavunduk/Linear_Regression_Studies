import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#Linar Regression için gereken modül LinearRegression

dataset = pd.read_csv('your data')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#X_train bağımsız değişkenlerin trainidir, y_train ise bağımlı değişkenlerin trainidir.
regressor = LinearRegression()
#Simple Liear Regressionda parantezin içine parametre girmen gerekmiyor diğer regressionlarda gerekebilir.
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
#y-PRED ÖNGÖRÜLEN MAAŞLAR, Y_TEST GERÇEK MAAŞLAR

#TRAINING SET VISUALIZING
plt.scatter(X_train, y_train, color = 'red')
#scatter red pointleri yerleştirir
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
#ilk iki alan koordinatları temsil ediyor, eksenlerde ne yazacağını giriyoruz. Bu bizim regression lineımız
plt.title('Salary vs Experience (Training set)')
#title ı koyduk
plt.xlabel('Years of Experience')
#x ekseni adı
plt.ylabel('Salary')
#y ekseni adı
plt.show()
#gösteriyor

#TEST SET VISUALIZING
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
#burada regression lineımız aynı olduğu için bir değişiklik yapmıyoruz. X_test olarak değiştirmeyeceğiz.
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

