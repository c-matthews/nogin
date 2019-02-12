import numpy as np 



class Integrator:
    
    def __init__( self,args ):
    
        self.dt = args.dt
        self.g = args.g
        self.acc = 0
        
        self.itype = args.i
        self.noginexact = args.noginexact
        self.nogincg = args.nogincg
        self.adaptsghmc = args.adaptsghmc
        self.nonoise = args.nonoise 
        self.autofriction = False
        self.RFriction = IgnoreFriction
        known = False
        
        if (self.dt is None):
            raise Exception('No dt specified')
        
        if (self.dt<0):
            raise Exception('Timestep cannot be negative')
        
        if (self.itype is None):
            raise Exception('No integrator specified')
        
            
        
        if (self.itype.lower()=='baoab'):
            known = True
            from integrators.baoab import Step, Setup, ResetFriction
            self.RFriction = ResetFriction
            self.autofriction = args.autofriction
            
        if (self.itype.lower()=='msgld'):
            known = True
            from integrators.msgld import Step, Setup
            
        if (self.itype.lower()=='ccadl'):
            known = True
            from integrators.ccadl import Step, Setup, ResetFriction
            self.RFriction = ResetFriction
            self.autofriction = args.autofriction
            
        if (self.itype.lower()=='sgnht'):
            known = True
            from integrators.sgnht import Step, Setup
            
        if (self.itype.lower()=='sgld'):
            known = True
            from integrators.sgld import Step, Setup
            
        if (self.itype.lower()=='mala'):
            known = True
            from integrators.mala import Step, Setup
            
        if (self.itype.lower()=='sghmc'):
            known = True
            from integrators.sghmc import Step, Setup
            
        if (self.itype.lower()=='nogin'):
            known = True
            from integrators.nogin import Step, Setup, ResetFriction
            self.RFriction = ResetFriction
            self.autofriction = args.autofriction

        if (not known):
            raise Exception('Unknown integrator specified: ' + self.itype.lower())

        self.IStep = Step
        self.ISetup = Setup


    def Step(self, ii ):
        self.m.q,self.m.p,self.extra = self.IStep( self, self.m.q , self.m.p, self.extra  )

        if ((ii>=0.05*self.output.N) and (self.doneautofriction is False)):
            self.DoAutoFriction()
            self.doneautofriction = True

        return

    def Setup(self,syst):
    
        rng,_,model,output = syst
        
        self.dt2 = self.dt/2.0
        self.dt4 = self.dt/4.0
        self.extra = None
        
        self.rng = rng
        self.m = model
        self.output = output
    
        self.ISetup(self)
        self.doneautofriction = False

    def GetForce(self,q,getcovariance=True,updatellh=True):

        llh,f,c = self.m.GetForce(q,getcovariance)

        if (updatellh):
            self.m.llh = llh

        if (getcovariance):
            return f , c
        else:
            return f


    def DoAutoFriction(self):
    
        if (not self.autofriction):
            return

        X = self.output.GetXSoFar()

        X = X[:,int(X.shape[1]*0.1):]

        covX = np.cov(X)
        
        try:
            eigw,eigv = np.linalg.eig( covX )
        except:
            print "  >> Auto Friction Failed!"
            return
        
        ff = 1.0 / np.sqrt( np.max( eigv.real ) )
        
        if (ff<1e-4):
            ff = 1e-4
        
        if (ff>1e4):
            ff = 1e4
        
        self.oldg = self.g
        self.g = ff
        
        print "  >> Auto Friction success, new friction = " + str(ff)
        
        self.RFriction(self)
        return





def IgnoreFriction( X ):
    print "  >> Friction ignored for this integrator"
    pass





        
