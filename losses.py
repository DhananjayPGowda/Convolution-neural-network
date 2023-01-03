import numpy as np
                                  
class MSE():
        def __init__(self):
            self.output = 'none'
            self.tot_loss = 0
        def loss(self,p,c):
            self.tot_loss = np.sum(((c-p)**2)/c.shape[-1]) /(c.shape[0])          
        def dow(self,p,c):
             self.cost = 2*(p-c)/c.shape[-1]




class Categorical_Crossentropy():
        def __init__(self):
            self.output = 'none'
            self.tot_loss = 0
        def loss(self,p,c):
            self.tot_loss = -np.sum((c*np.log(p)))/(c.shape[0]*c.shape[-1])
            #return self.tot_loss
        def dow(self,p,c):         
            if type(self.output).__name__ == 'Softmax':
                self.cost =  p - c
            else:
                self.cost = ((-c/p) + ((1-c)/(1-p)))
