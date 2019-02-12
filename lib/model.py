import numpy as np
import scipy.io as sio
import os

class Model:
    
    def __init__( self,args ):

        if (args.m is None):
            raise Exception('No model specified')

        if (args.d is None):
            raise Exception('No dataset specified')

        self.model = args.m.lower()
        known = False
        
        self.dataset = args.d
        self.bsize_int = args.bsize
        self.bsize_pc = args.bsize_pc
        self.UseExactCov = args.exactcovariance
        self.UseVariance = args.diagonalcovariance
        self.CC = None
        self.CLambda = args.clambda
        self.torchload = args.torchload
        
        if (args.covhist>0):
            self.covhist_len = args.covhist
            self.covhist_ii  = 0
            self.covhist_do  = True
        else:
            self.covhist_do  = False
        
        
        if (self.model.lower()=='gmean'):
            known = True
            from models.gmean import LLH, Setup
        
        if (self.model.lower()=='blr'):
            known = True
            from models.blr import LLH, Setup
        
        if (self.model.lower()=='ica'):
            known = True
            from models.ica import LLH, Setup
        
        if (self.model.lower()=='nn'):
            known = True
            from models.nn import LLH, Setup

        if (not known):
            raise Exception('Unknown model specified: ' + self.model.lower())
            
        self.MSetup = Setup
        self.MLLH = LLH
        


    def Setup( self, syst ):
        rng,ig,_,output = syst

        self.rng = rng
        self.ig = ig
        self.output = output

        print "Loading dataset..."
        if (not self.torchload):
            if (not os.path.isfile(self.dataset) ):
                raise Exception('Unable to find data: ' + self.dataset)
            self.data = sio.loadmat( self.dataset )
            self.q = np.array(self.data['ic']).astype('float64')
        else:
            import torchvision
            import torch
            self.data = torchvision.datasets.MNIST(root=self.dataset, train=True,
                    download=True, transform=torchvision.transforms.ToTensor())
            self.DataSize = 1000
            mysampler = torch.utils.data.SubsetRandomSampler(xrange(self.DataSize))
            self.q = np.loadtxt(self.dataset + '/ic.txt')[:,np.newaxis]
        
        print "Data loaded OK"

        self.p = self.rng.randn( self.q.shape[0], self.q.shape[1] )
        self.llh = 0
        self.Cfac = self.CLambda
        
        self.ndof = self.q.size
        
        if (self.covhist_do):
            self.covhist = np.zeros( ( self.covhist_len , self.q.shape[0]  ))
        
        self.MSetup(self)
        
        self.k = 0
        if (self.bsize_int>0):
            self.k = self.bsize_int
        if (self.bsize_pc>0):
            self.k = int(self.bsize_pc * self.DataSize)
        if (self.k<1):
            self.k = 1
        if (self.k>self.DataSize):
            self.k = self.DataSize
        
        print "Batchsize: " + str(self.k) + " of " + str(self.DataSize) + ";   System size: " + str(self.q.shape)
        
        if (self.torchload):
            self.traindata = torch.utils.data.DataLoader(self.data, batch_size=self.k,
                                          shuffle=False,sampler=mysampler)
        
        self.InitForce(self.q )
        


    def GetForce(self, q, getcovariance=True ):

        idxs = self.rng.choice( xrange( self.DataSize) , self.k, replace=False )
        kfac = self.DataSize / (1.0 * self.k)
        kfacllh = kfac

        if (self.UseExactCov and getcovariance):
            llh, fall, plh, pf  = self.MLLH( self, q , xrange(self.DataSize) )
            f = fall[:,idxs]
            kfacllh = 1.0
        else:
            llh, f, plh, pf  = self.MLLH( self, q , idxs )

 
        F = pf+kfac*np.sum(f,axis=1,keepdims=True)
        V = plh+kfacllh*llh
        
        if (getcovariance and self.covhist_do):
            self.covhist[self.covhist_ii,:] = (F-pf).flatten()
            self.covhist_ii = (self.covhist_ii +1 ) % self.covhist_len
            return V, F, self.covhist 
        
        if (getcovariance and (self.k<self.DataSize)):
            
            if (self.UseExactCov):
                if (self.UseVariance):
                    np.diag(np.var(fall,axis=1,ddof=1))
                else:
                    C = np.cov( fall )
            else:
                if (self.UseVariance):
                    np.diag(np.var(f,axis=1,ddof=1))
                else:
                    C= np.cov( f )
            
            C = C * ( (self.DataSize - self.k ) * kfac  )
            
            self.CC = self.Cfac * C + (1-self.Cfac)*self.CC

        return V , F, self.CC



    def InitForce(self, q ):
    
        llh, fall, plh, pf  = self.MLLH( self, q , xrange(self.DataSize) )

        kfac = 1.0
 
        F = pf+kfac*np.sum(fall,axis=1,keepdims=True)
        V = plh+kfac*llh
        self.llh = V
        self.F = F
        
        if (self.covhist_do):
            for ii in xrange(self.covhist_len):
                aa,bb,cc = self.GetForce( self.q )
            return
        
        C = np.cov( fall )
        C = C * ( (self.DataSize - self.k ) * (self.DataSize / (1.0 * self.k))  )
        
        self.CC =  C
        

        return

    def CovVec(self, C , v ):

        if (self.covhist_do):
            #cc = self.covhist - np.mean(self.covhist,axis=0)
            cc = C - np.mean(C,axis=0)
            res = np.dot( cc.T, np.dot(cc,v)) / (self.covhist_len - 1)
            return res
        else:
            return np.dot( C , v )
