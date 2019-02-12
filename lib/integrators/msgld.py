import numpy as np

def Step(IG, q,p,extra):
 
    dt = IG.dt
    rng = IG.rng
    
    #   (force update)
    f, C = IG.GetForce(q)
    
    # Random number
    R = rng.randn( p.shape[0], p.shape[1] )
    
    # Update position
    q += IG.h * f + dt*R
    q -= IG.h * dt * IG.m.CovVec(C,R) #np.dot( C , R )
    
    return q , p , extra



def Setup( IG ):

    print("Using mSGLD integrator")
    print(" : Vollmer, Zygalakis and Teh, JMLR (2016)")
    print(" : see https://arxiv.org/abs/1501.00438")
 

    IG.extra = []
    IG.h = IG.dt * IG.dt * 0.5





