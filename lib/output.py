import numpy as np
from datetime import timedelta
from time import time
import datetime
import scipy.io as sio
import pandas as pd

class Output:
    
    def __init__( self,args ):

        if (args.o is None):
            raise Exception('No output location specified')
        
        self.o = args.o
        self.N = args.n
        
        self.ofreq = args.ofreq
        self.pfreq = args.pfreq
        self.tfreq = args.tfreq
        if (self.pfreq==0):
            self.pfreq = 10
        self.pfreq = (self.N // self.pfreq)
        self.burn_pc = args.burn_pc
        
        self.savetrajectory = args.savetrajectory
        self.skipenergy = args.skipenergy
        self.hbins = args.histogrambins
        self.ranum = args.rollingaverages
        self.acf = args.autocorrelationfn
            



    def Setup( self, syst ):
        rng,ig,model,_ = syst

        self.rng = rng
        self.ig = ig
        self.model = model
    
        self.stepcount = 0
        self.ocount = 0
    
        self.TotalSaves = 1 + self.N // self.ofreq


        self.X = np.zeros( (self.model.ndof, self.TotalSaves) )
        if (self.skipenergy):
            self.E = 0
        else:
            self.E = np.zeros( (2 , self.TotalSaves) )
 
        self.lastupd = time()
        self.starttime = time()

    def Save(self):
    
    
        ctime = datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S')
        timetaken = time()-self.starttime
        print("Time elapsed: " + str(timetaken) + ".")
        
        self.X = self.X[:,:self.ocount]
        if (not self.skipenergy):
            self.E = self.E[:,:self.ocount]
        
        savedata = {'E':self.E,'N':self.N,'T':timetaken,'date':ctime,'dt':self.ig.dt,'g':self.ig.g,\
            'ig':self.ig.itype,'model':self.model.model,'k':self.model.k,'dataset':self.model.dataset,\
            'ofreq':self.ofreq,'acc':self.ig.acc/(1.0*self.N),'cfac':self.model.Cfac}
        
        bstart = int(self.X.shape[1] * self.burn_pc)
        self.Xb = self.X[:,bstart:]
        savedata['Xmean'] = np.mean(self.Xb,axis=1)
        savedata['Xcov'] = np.cov(self.Xb)
        
        
        
        if (self.savetrajectory):
            savedata['X'] = self.X

        if (self.ranum>0):
            rollingmean, rollingvar, rollingmean2, rollingvar2, rollingt  = self.RollingAverages()
            savedata['Rmean'] = rollingmean
            savedata['Rvar']  = rollingvar
            savedata['Rmean2'] = rollingmean2
            savedata['Rvar2']  = rollingvar2
            savedata['Rt'] = rollingt
            
        if (self.hbins>0):
            Hy = np.zeros( (self.Xb.shape[0], self.hbins ))
            Hx = np.zeros( (self.Xb.shape[0], self.hbins+1 ))
            for ii in range( self.Xb.shape[0] ):
                aa,bb = np.histogram( self.Xb[ii,:], bins=self.hbins, density=True)
                Hy[ii,:] = np.copy(aa)
                Hx[ii,:] = np.copy(bb)
            savedata['Hy'] = Hy
            savedata['Hx'] = Hx
            
        if (self.acf>0):
            acf = np.zeros( (self.Xb.shape[0], self.acf ))
            print("Computing autocorrelation functions...")
            tm = time()
            st = time()
            for ii in range( self.Xb.shape[0] ):
                xx=(self.Xb[ii,:])
                xx=xx-np.mean(xx)
                xx2 = np.mean(xx**2)
                acf[ii,0]=1
                for jj in xrange( self.acf-1 ):
                    jjj=jj+1
                    acf[ii,jjj] = np.mean( xx[jjj:] * xx[:-jjj])/xx2
                if (time()-tm>10):
                    print ("   >> On " + str(ii) + " of " + str(self.Xb.shape[0]) +  ". Time: " + str(time()-st))
                    tm = time()
            savedata['acf'] = acf

        
        
        

        print("Saving output to " + self.o)
        sio.savemat( self.o , savedata )
        print("Saved OK")

    def Log(self):

        self.stepcount += 1

        if ((self.stepcount % self.ofreq)==0 ):
            self.X[ :, self.ocount ] = np.squeeze(np.copy( self.model.q ))
            if (not self.skipenergy):
                self.E[ 0, self.ocount ] = self.model.llh
                self.E[ 1, self.ocount ] = np.sum(self.model.p**2)*0.5
            self.ocount += 1

        if ((self.stepcount % self.pfreq)==0 ) or ((time() - self.lastupd)>self.tfreq):
            self.lastupd = time()
            dt = time()-self.starttime
            rt = dt * (self.N / (1.0*self.stepcount)-1.0)
            if (self.ig.itype=='mala'):
                accstr = '  ::  acc=' + str( self.ig.acc / (1.0*self.stepcount))
            else:
                accstr = ''
            
            print(" Step " + str(self.stepcount) + " of " + str(self.N) + " (" + str(int( (1000.0*self.stepcount)/self.N)/10.0) \
                + "%)  ::  Elapsed=" + str(timedelta(seconds=int(dt))) + " Remaining=" + str(timedelta(seconds=int(rt)))\
                + "  ::   llh = " + str(self.model.llh ) + accstr)



    def GetXSoFar(self):

        return self.X[:,:self.ocount]

    def RollingAverages(self):
    
        print("Computing rolling averages...")
        tm = time()
        st = time()

        ranum = self.ranum
        X = self.Xb

        N = X.shape[1]
        D = X.shape[0]

        Rmean = np.zeros((D,ranum))
        Rvar = np.zeros((D,ranum))
        Rmean2 = np.zeros((D,ranum))
        Rvar2 = np.zeros((D,ranum))

        t = np.around(np.geomspace( 10 , N , num=ranum )).astype(int)
        
        df = pd.DataFrame( X.T  )
        
        for ii,tt in enumerate(t):
            rv = df.rolling(tt)
            rvmean = rv.mean().values
            rvvar = rv.var().values
            Rmean[:,ii] = np.nanmean( (rvmean), axis=0 )
            Rvar[:,ii] = np.nanmean( (rvvar), axis=0 )
            Rmean2[:,ii] = np.nanmean( (rvmean)**2, axis=0 )
            Rvar2[:,ii] = np.nanmean( (rvvar)**2, axis=0 )
            if (time()-tm>10):
                print("   >> On " + str(ii) + " of " + str(ranum) +  ". Time: " + str(time()-st))
                tm = time()


        return  Rmean, Rvar, Rmean2, Rvar2, t

        
