    
import numpy as np
from activations import *
from layers import *
from optimizers import *
from models import *
from losses import *
import random as rd,pickle

with open('mnist_original.pkl','rb') as f:
    data = pickle.load(f)
    X = data['X']/255
    Y = data['Y']
    x = data['x']/255
    y = data['y']
    
    
X = np.reshape(X,(len(X),1,28,28))
x = np.reshape(x,(len(x),1,28,28))


nn = NN()
nn.add(Input((1,28,28)))
nn.add(Conv2D(16,(3,3)))
nn.add(Relu())
nn.add(Conv2D(32,(3,3)))
nn.add(Relu())
nn.add(Flatten())

nn.add(Dense(10))
nn.add(Softmax())
nn.compile(optimizer = Adam(lr = .001),loss = Categorical_Crossentropy())
s  = 0
v = None

nn.fit(X[s:v],Y[s:v],epochs = 1,batch_size = 32,validation_data = (x, y), verbose = 1)

P = np.round(nn.predict(X[:10]))
for e,i in enumerate(P[:10]):
    print(i,Y[e])
