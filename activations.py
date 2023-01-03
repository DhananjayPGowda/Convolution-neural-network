import numpy as np
class Yolo():
    def __init__(self):
      self.sig = Sigmoid()
    def predict(self):
        self.res = np.copy(self.prev.res)

   
        self.res[:,:-4,:,:] = self.sig.predict(self.prev.res[:,:-4,:,:],1)
        if not self.is_output:
            self.next.predict()
    def initialize(self):
        self.res = np.empty_like(self.prev.res)
    def back_prop(self):
        self.cost = np.copy(self.next.cost)
        self.cost[:,:-4,:,:] = self.res[:,:-4,:,:]*(1-self.res[:,:-4,:,:])*self.next.cost[:,:-4,:,:]
        self.prev.back_prop()

#class Sigmoid():
#     
#    def predict(self):
#        self.res = 1/(1+np.exp(-self.prev.res))
#        if not self.is_output:
#            self.next.predict()
#    def initialize(self):
#        self.res = np.empty_like(self.prev.res)
#    def back_prop(self):
#        self.cost = self.res *(1-self.res)*self.next.cost
#        self.prev.back_prop()





class Sigmoid():
     
    def predict(self,P = None,action= 0):
        if action:
           return 1/(1+np.exp(-P))
        self.res = 1/(1+np.exp(-self.prev.res))
        if not self.is_output:
            self.next.predict()

    def initialize(self):
        self.res = np.empty_like(self.prev.res)
    def back_prop(self):
        self.cost = self.res *(1-self.res)*self.next.cost
        self.prev.back_prop()

        
        
class Relu():
    def initialize(self):
        self.res = np.empty_like(self.prev.res)
    def predict(self):
        self.res = np.where(self.prev.res < 0,0,self.prev.res)
        self.p_shape = self.res.shape
        if not self.is_output:
            self.next.predict()
    def back_prop(self):
            #if self.res.shape != self.next.cost.shape:
#                self.next.cost = np.reshape(self.next.cost,(self.res.shape))
            self.cost = np.where(self.res<=0,0, self.next.cost)
            self.prev.back_prop()
            
class Softmax():
    def initialize(self):
        self.res = np.zeros_like(self.prev.res)
    def predict(self):            
        e = np.exp(self.prev.res)
        s = np.sum(e,axis = 1)#+.00000001
        self.res = e/s[:,None]
        if not self.is_output:
            self.next.predict()
    def back_prop(self):
        if (self.is_output) and type(self.next).__name__ == 'Categorical_Crossentropy':
            self.cost = np.copy(self.next.cost)
        else:
            self.cost = self.res*(1-self.res)*self.next.cost
        self.prev.back_prop()

class Reshape():
    def __init__(self,shape):
        self.shape = list(shape)
    def initialize(self):
        self.res = np.zeros(self.shape)
    def predict(self):
        s = self.shape
        s.insert(0,self.prev.res.shape[0])
        self.res = np.reshape(self.prev.res,(tuple(s)))
        s.pop(0)
        if not self.is_output:
            self.next.predict()
    def back_prop(self):
        self.cost = np.reshape(self.next.cost,(self.prev.res.shape))
        self.prev.back_prop()              