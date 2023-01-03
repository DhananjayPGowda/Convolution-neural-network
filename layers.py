
import numpy as np,math
from time import time



class Base():
    
    
    def summary(self):
        try:
            print(type(self).__name__,self.res.shape)
        except:
            pass
        try:
            self.next.summary()
        except:
            pass
            
                                  
class Input(Base):
    def __init__(self,shape):
        self.res = np.random.random(shape)
        
        
    def predict(self,INPUT):
        self.res = INPUT
        self.next.predict()
    '''dummy methods'''
    
    
    def initialize(self):
        return
        
        
    def back_prop(self):
        return

        
class Dense(Base):
    def __init__(self,p):
        self.p = p
        
        
    def initialize(self):
        self.W = np.random.randn(self.p,self.prev.res.shape[-1]) * np.sqrt(2/self.prev.res.shape[-1])
        self.B = np.zeros((self.p,1))
        self.res = np.empty((1,self.p))
        
        
    def predict(self):
            self.res = np.dot(self.W,self.prev.res.T)
            self.res  = (self.res+ self.B).T
            self.next.predict()
            
        
    def back_prop(self):
         self.next.cost = self.next.cost.T
         self.cost = np.dot(self.W.T,self.next.cost).T      
         self.updw = np.dot(self.prev.res.T,self.next.cost.T).T         
         self.updb = np.sum(self.next.cost,axis = 1)
         self.updb = np.reshape(self.updb,(len(self.updb),1))
         self.optimizer.update(self.W,self.B,self.updw,self.updb)        
         self.prev.back_prop()
         
         
class Conv2D(Base):
    
    
    def __init__(self,n,kernel = (2,2),strides=(1,1)):
        self.n = n
        self.strip = strides
        self.kernel_size = kernel
        
        
    def initialize(self):       
        shape = (self.n,)+tuple((np.array(self.prev.res.shape[1:])-np.array(self.kernel_size)+1)/np.array(self.strip))
        self.output_shape = np.ceil(np.array(shape)).astype('int64')
        self.res_shape = tuple(self.output_shape)
        self.res = np.zeros((self.output_shape))        
        self.bias = np.zeros((self.output_shape))        
        self.bs = self.prev.res.shape[0]
        self.kernels = np.random.rand(self.n,self.kernel_size[0]*self.kernel_size[1]*self.bs)/10


    def convolve(self,img):
        conv= []
        for y in range(0,(img.shape[1]-img.shape[1]%self.strip[0])-self.kernel_size[0]+1,self.strip[0]):
            for x in range(0,(img.shape[2]-img.shape[2]%self.strip[1])-self.kernel_size[1]+1,self.strip[1]):
                conv.append(img[:,y:y+self.kernel_size[0],x:x+self.kernel_size[1]].flatten())
        conv = np.array(conv).T
        return conv

                
    def rev_convolve(self,conv):
        img= np.zeros_like(self.pre_conv)
        count = 0
        for y in range(0,(img.shape[1]-img.shape[1]%self.strip[0])-self.kernel_size[0]+1,self.strip[0]):
            for x in range(0,(img.shape[2]-img.shape[2]%self.strip[1])-self.kernel_size[1]+1,self.strip[1]):
                img[:,y:y+self.kernel_size[0],x:x+self.kernel_size[1]] += np.reshape(conv[:,count],((img.shape[0],)+self.kernel_size))
                count += 1
        return np.array(img)


    def predict(self):
        ps = self.prev.res.shape
        f = self.prev.res.flatten()        
        self.pre_conv = np.reshape(f,((ps[0]*ps[1],)+ps[2:] ))         #combine multiple images into single image
        conv = self.convolve(self.pre_conv)        
        self.conv = np.array_split(conv,ps[0])             #split comined convoluted images into corresponding images
        res = np.matmul(self.kernels,self.conv)
        self.dot_shape = res[0].shape
        self.res = np.reshape(res,(res.shape[0],)+self.res_shape)
        self.res += self.bias                          
        self.next.predict()
                  

    def back_prop(self):
        self.updb = np.sum(self.next.cost,axis = 0)
        self.next.cost  = np.reshape(self.next.cost,(len(self.next.cost),)+self.dot_shape)
        back_conv = np.array(self.conv).transpose(0,2,1)
        self.updw = np.sum(np.matmul(self.next.cost,back_conv),axis = 0)        
        cost = np.matmul(self.kernels.T,self.next.cost)
        l = cost.shape[0]
        cost = np.reshape(cost,(cost.shape[0]*cost.shape[1],cost.shape[2]))        
        cost = self.rev_convolve(cost)
        self.cost = np.array_split(cost,l)            
        self.optimizer.update(self.kernels,self.bias,self.updw,self.updb)        
        self.prev.back_prop()



        
class MaxPooling2D(Base):
    def __init__(self,size = (2,2),strides = (2,2)):
        self.size = size
        self.strides = strides


    def initialize(self):
        shape = (self.prev.res.shape[0],math.ceil(self.prev.res.shape[1]/self.strides[0]),math.ceil(self.prev.res.shape[2]/self.strides[1]))
        self.res = np.zeros(shape)
        
            
    def predict(self):
        res = []
        for i in self.prev.res:
            pool = []
            for y in range(0,i.shape[1],self.strides[0]):
                for x in range(0,i.shape[2],self.strides[1]):
                    pi = i[:,y:y+self.size[0],x:x+self.size[1]]
                    Max = np.max(np.max(pi,axis = 2),axis = 1)
                    pool.append(Max)
            pool = np.array(pool)
            res.append(np.reshape(pool.T,(i.shape[0],math.ceil(i.shape[1]/self.strides[0]),math.ceil(i.shape[2]/self.strides[1]))))
        self.res = np.array(res)
        self.next.predict()


    def back_prop(self):
        cost = np.zeros_like(self.prev.res)
        for e,i in enumerate(self.next.cost):
            for z in range(i.shape[0]):
                for y in range(i.shape[1]):
                    for x in range(i.shape[2]):
                        I = self.prev.res[e][z,y*self.strides[0]:(y*self.strides[0])+self.size[0],x*self.strides[1]:(x*self.strides[1])+self.size[1]]
                        w= np.where(I==self.res[e][z,y,x])
                        l = len(w[0])
                        W = list(w)
                        W[0] += y*self.strides[0]
                        W[1] += x*self.strides[1]
                        W = tuple(W)
                        cost[e,z][W] = i[z,y,x] / l
        self.cost = cost
        self.prev.back_prop()                        
                
                                
       
                 
class Flatten(Base):
    def initialize(self):
        self.in_shape  = self.prev.res.shape
        self.res = np.zeros((1,len(self.prev.res.flatten()))) 
        
        
    def predict(self):
        self.in_shape = self.prev.res.shape
        self.res = np.reshape(self.prev.res,(self.in_shape[0],math.prod(self.in_shape[1:])))
        self.next.predict()
        
        
    def back_prop(self):
        self.cost = np.reshape(self.next.cost,self.in_shape)
        self.prev.back_prop()        