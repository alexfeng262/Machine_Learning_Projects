import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

def plot(x,y):
    plt.figure()
    plt.title('Precio de pizza vs diametro')
    plt.xlabel('Diametro en pulgadas')
    plt.ylabel('Precio en dolar')
    plt.plot(x,y,'k.') # 'k.' indica la caracteristica del punto. En este caso k = negro y .= que sea un punto
    plt.axis([0,25,0,25])
    plt.grid(True)
    plt.show()

#training set
pizza_diameter_train = np.array([6,8,10,14,18]).reshape(-1,1)
pizza_price_train = np.array([7,9,13,17.5,18])

#test set
pizza_diameter_test = np.array([8,9,11,16,12]).reshape(-1,1)
pizza_price_test = np.array([11,8.5,15,18,11])

#The model
model = LinearRegression()
model.fit(pizza_diameter_train, pizza_price_train)

test_pizza = np.array([[12]])

predict_price = model.predict(test_pizza)[0]
score = model.score(pizza_diameter_test,pizza_price_test)
print( '%.2f' % score)

plot(pizza_diameter_test,pizza_price_test)



