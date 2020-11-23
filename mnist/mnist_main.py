import pandas as pd     
import numpy as np 
from tensorflow.keras.datasets.mnist import load_data
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
import time

# (x_train,y_train),(x_test,y_test) = load_data(path='mnist.npz')
# data = load_data(path='mnist.npz');np.savez_compressed('mahmood.npz', data) // downloading for testing 
(x_train,y_train),(x_test,y_test) = load_data(path='mahmood.npz')
print(x_train.shape)
x_train = x_train/255
x_test = x_test/255
x_train = np.reshape(x_train,(60000,784))
x_test = np.reshape(x_train,(60000,784))
print(x_train.shape)


# t1 = time.process_time()
# knn = KNeighborsClassifier(n_neighbors=10)
# knn.fit(x_train,y_train)
# prediction = knn.predict(x_test)
# accuracy = accuracy_score(prediction,y_test)
# print(accuracy)
# t2 = time.process_time()
# print(t2-t1)


t1 = time.process_time()
sv = svm.SVC()
sv.fit(x_train,y_train)
prediction = sv.predict(x_test)
accuracy = accuracy_score(prediction,y_test)
t2 = time.process_time()
print(t2-t1)









