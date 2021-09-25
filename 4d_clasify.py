import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf

np.random.seed(1233231)
x1 = np.random.random(5000)
x2 = np.random.random(5000)
print(x1[0:10])
print(x2[0:5])

plt.figure(figsize=(7,7))
plt.plot(x1,x2,'y.')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.show()


def classify_data(x1, x2):
    target = []
    for i in range(len(x1)):
        if x1[i] < 0.5 and x2[i] < 0.5:
            target.append('type1')
        elif x1[i] >= 0.5 and x2[i] < 0.5:
            target.append('type2')
        elif x1[i] < 0.5 and x2[i] >= 0.5:
            target.append('type3')
        elif x1[i] >= 0.5 and x2[i] >= 0.5:
            target.append('type4')
    return np.array(target)


target = classify_data(x1, x2)
print(target[0:10])

print(len(target[target == 'type1']))
print(len(target[target == 'type2']))
print(len(target[target == 'type3']))
print(len(target[target == 'type4']))


plt.figure(figsize=(7,7))
plt.plot(x1[target == 'type1'], x2[target == 'type1'], 'b.')
plt.plot(x1[target == 'type2'], x2[target == 'type2'], 'r.')
plt.plot(x1[target == 'type3'], x2[target == 'type3'], 'g.')
plt.plot(x1[target == 'type4'], x2[target == 'type4'], 'y.')
plt.xlabel('x1')
plt.ylabel('y1')
plt.grid(True)
plt.show()

def merge_data(x1,x2):
    ret = []
    for i in range(len(x1)):
        ret.append([x1[i], x2[i]])
    return np.array(ret)

data = merge_data(x1, x2)
data[0:5]

encoder = LabelBinarizer()
print(target[0:10])
target = encoder.fit_transform(target)
print(target[0:10])
print(encoder.classes_)


x_train = data[0:4000]
t_train = target[0:4000]

x_test = data[4000:]
t_test = target[4000:]

model1 = Sequential()
model1.add(Dense(16, input_dim = 2, activation = 'sigmoid'))
model1.add(Dense(16, activation = 'sigmoid'))
model1.add(Dense(4,activation = 'softmax'))

model1.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

hist1 = model1.fit(x_train, t_train, epochs = 50, batch_size = 20)


