from sklearn import datasets, linear_model
import numpy as np
import matplotlib.pyplot as plt


diabete_dataset = datasets.load_diabetes()
#Usar una sola caracteristica (feature)
diabete_X = diabete_dataset.data[: np.newaxis,2].reshape(-1,1)

diabete_X_train = diabete_X[:-20]
diabete_X_test = diabete_X[-20:]

diabete_Y_train = diabete_dataset.target[:-20]
diabete_Y_test = diabete_dataset.target[-20:]

model = linear_model.LinearRegression()
model.fit(diabete_X_train,diabete_Y_train)

print(model.score(diabete_X_test,diabete_Y_test))
