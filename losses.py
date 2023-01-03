import numpy as np

      
class Yolo_loss():
        def __init__(self):
            self.output = 'none'
            self.tot_loss = 0
            self.cat = Categorical_Crossentropy()
            self.mse= MSE()
        def loss(self,p,c):
            self.tot_loss = np.sum(((c-p)**2)/len(c))           
        def dow(self,p,c):
             self.cost = np.zeros_like(p)
             self.cat.dow(p[:,:-4,:,:],c[:,:-4,:,:])
             self.cost[:,:-4,:,:] = self.cat.cost
             for i in range(p.shape[0]):
                for xx in range(p.shape[2]):
                   for yy in range(p.shape[2]):
                      if c[i,0,xx,yy] == 1:
                        self.mse.dow(p[i,-4:,xx, yy],c[i,-4:,xx,yy])
                
                        
                        self.cost[i,-4:,xx, yy] = self.mse.cost
                        
                                  
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