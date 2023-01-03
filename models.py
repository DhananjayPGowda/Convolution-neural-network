
import numpy as np,sys,copy
from time import time
from optimizers import Adam




class NN():
    def __init__(self):
        self.layers = []
    def accuracy(self,p,y):
        P = np.round(p)
        P = np.sum((y - P)**2,axis = 1)
        P =len(y)- len(np.where(P>0)[0])
        return P/len(y)
    def get_loss_and_accuracy(self,vX,vY):
        size = 1000
        replace = False
        if len(vX)<size:
            replace = True
        rdm = np.random.choice(len(vX), size= size, replace=replace)
        vX = vX[rdm,:]
        vY = vY[rdm,:]
        P = self.predict(vX)
        self.loss.loss(P,vY)
        vl = self.loss.tot_loss
        va = self.accuracy(P,vY)
        vl = round(vl,5)
        va = round(va,4)
        return vl,va
    def add(self,l):
        try:
            i = self.layers
            try:
                while 1:
                    i = i.next
            except:
                l.prev = i
                l.initialize()
                i.next = l
                
        except:
            self.layers = l
            self.input = l
            
        self.output = l
        l.is_output = 1
        try:
            l.prev.is_output = 0
        except:
            pass
    def fit(self,X,Y,batch_size = 64,epochs = 1,validation_data = None,verbose = 1):#
        for epoch in range(epochs):
            #print('epoch : ',epoch)
            for batch in range(0,X.shape[0]-batch_size,batch_size):
                #print('batch : ',batch)
                x = X[batch:batch_size+batch]
                y = Y[batch:batch_size+batch]
                P = self.predict(x)
                self.loss.dow(P,y)
                if verbose:
                    self.loss.loss(P,y)                    
                    accuracy = np.round(self.accuracy(P,y),4)
                    loss = round(self.loss.tot_loss,5)
                    sys.stdout.write('\r epoch : '+str(epoch)+' batch : '+str(batch)+' loss. :  '+str(loss)+' accuracy : '+str(accuracy))
                else:
                    sys.stdout.write('\r epoch : '+str(epoch)+' batch : '+str(batch))
                self.output.back_prop()
            if verbose == 1:
                l,a= self.get_loss_and_accuracy(X,Y)
                sys.stdout.write('\r epoch '+str(epoch)+' loss : '+str(l)+ ' accuracy : '+str(a))     
                try:
                    if validation_data != None:
                        vX = validation_data[0]
                        vY= validation_data[1]
                        vl,va = self.get_loss_and_accuracy(vX,vY)
                        sys.stdout.write(' val_loss : '+str(vl)+' val_accurracy : '+str(va)+'\n')
                    else:
                        print()
                except:
                    print()
    def predict(self,INPUT):
            self.input.predict(INPUT)
            return self.output.res
    def compile(self,loss,optimizer = Adam()):
        self.loss = loss
        self.output.next = loss
        self.loss.output = self.output
        l = self.input
        while(not l.is_output):
           l.optimizer = copy.copy(optimizer)
           l = l.next