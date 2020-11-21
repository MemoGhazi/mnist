import pandas as pd     
import numpy as np 
from tensorflow.keras.datasets.mnist import load_data

# (x_train,y_train),(x_test,y_test) = load_data(path='mnist.npz')

# data = load_data(path='mnist.npz') // downloading for testing 
# np.savez_compressed('mahmood.npz', data)
(x_train,y_train),(x_test,y_test) = load_data(path='mahmood.npz')
print(x_train)



















