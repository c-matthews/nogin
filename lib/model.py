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
        
        if (args.covhist>0):
            self.covhist_do  = True
            self.Clist_len = args.covhist
            self.CList = []
        else:
            self.covhist_do  = False
        
        
        if (self.model.lower()=='gmean'):
            known = True
            from lib.models.gmean import LLH, Setup
        
        if (self.model.lower()=='blr'):
            known = True
            from lib.models.blr import LLH, Setup
        
        if (self.model.lower()=='ica'):
            known = True
            from lib.models.ica import LLH, Setup
        
        if (self.model.lower()=='gmix'):
            known = True
            from lib.models.gmix import LLH, Setup

        if (not known):
            raise Exception('Unknown model specified: ' + self.model.lower())
            
        self.MSetup = Setup
        self.MLLH = LLH
        


    def Setup( self, syst ):
        rng,ig,_,output = syst

        self.rng = rng
        self.ig = ig
        self.output = output

        print("Loading dataset...")
        if (not os.path.isfile(self.dataset) ):
            raise Exception('Unable to find data: ' + self.dataset)
        self.data = sio.loadmat( self.dataset )
        self.q = np.array(self.data['ic']).astype('float64')
        
        print("Data loaded OK")

        self.p = self.rng.randn( self.q.shape[0], self.q.shape[1] )
        self.llh = 0
        self.Cfac = self.CLambda
        
        self.ndof = self.q.size
        
        
        self.MSetup(self)
        
        self.k = self.DataSize
        if (self.bsize_int>0):
            self.k = self.bsize_int
        if (self.bsize_pc>0):
            self.k = int( np.round(self.bsize_pc * self.DataSize) )
        if (self.k<1):
            self.k = 1
        
        print("Batchsize: " + str(self.k) + " of " + str(self.DataSize) + ";   System size: " + str(self.q.shape))

        
        self.InitForce(self.q )
         

    def GetForce(self, q, getcovariance=True ):

        idxs = self.rng.choice( range( self.DataSize) , self.k, replace=False )
        kfac = self.DataSize / (1.0 * self.k)
        kfacllh = kfac

        if (self.UseExactCov and getcovariance):
            llh, fall, plh, pf  = self.MLLH( self, q , range(self.DataSize) )
            f = fall[:,idxs]
            kfacllh = 1.0
        else:
            llh, f, plh, pf  = self.MLLH( self, q , idxs )

 
        F = pf+kfac*np.sum(f,axis=1,keepdims=True)
        V = plh+kfacllh*llh
        
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
            
            self.CC = self.AppendClist(C) #self.Cfac * C + (1-self.Cfac)*self.CC

            #print(self.CC/( (self.DataSize - self.k ) * kfac  ))
        return V , F, self.CC



    def InitForce(self, q ):
    
        llh, fall, plh, pf  = self.MLLH( self, q , range(self.DataSize) )

        kfac = 1.0
 
        F = pf+kfac*np.sum(fall,axis=1,keepdims=True)
        V = plh+kfac*llh
        self.llh = V
        self.F = F
        
        C = np.cov( fall )
        C = C * ( (self.DataSize - self.k ) * (self.DataSize / (1.0 * self.k))  )
        
        self.CC =  C
        
        if (self.covhist_do):
            self.CList = [np.copy(C)] * self.Clist_len
    
        return
        
    def AppendClist( self, C ):
        if (not self.covhist_do):
            return C
    
        self.CList = [np.copy(C)] + self.CList[:-1]

        lam = 1.0
        sumlam = 0.0
        tC = 0
        for ii in range(len(self.CList)):
            tC += lam  * self.CList[ii]
            sumlam += lam
            lam = lam * self.Cfac
        
        return tC / sumlam


    def CovVec(self, C , v ):

        return np.dot( C , v )
#        if (self.covhist_do):
#            #cc = self.covhist - np.mean(self.covhist,axis=0)
#            cc = C - np.mean(C,axis=0)
#            res = np.dot( cc.T, np.dot(cc,v)) / (self.covhist_len - 1)
#            return res
#        else:
#            return np.dot( C , v )
