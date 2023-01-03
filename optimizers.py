import numpy as np

class SGD():
    def __init__(self,lr = 0.001):
        self.lr = lr
        
        
    def update(self,W,B,dw,db):
        W -= dw * self.lr
        B-= db *self.lr

        
class Adam():
    def __init__(self,lr = 0.001,b1 = 0.9,b2 = 0.9999,sigma = 10e-8):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.vb = 0
        self.mb = 0
        self.mw = 0
        self.vw = 0
        self.sigma = sigma
        
        
    def update(self,W,B,dw,db):
        self.mw = (self.b1*self.mw) + (1-self.b1)*dw
        self.vw = (self.b2*self.vw) + (1-self.b2)*(dw**2)
        self.mb = (self.b1*self.mb) + (1-self.b1)*db
        self.vb = (self.b2*self.vb) + (1-self.b2)*(db**2)
        mwh =  self.mw/(1-self.b1)
        vwh = self.vw/(1-self.b2)
        mbh = self.mb/(1-self.b1)
        vbh = self.vb/(1-self.b2)
        W -= (self.lr/(np.sqrt(vwh)+self.sigma)) * mwh
        B -= np.reshape((self.lr/(np.sqrt(vbh)+self.sigma)) * mbh,(B.shape))
