import numpy as np

def Step(IG, q,p,extra):
 
    dt = IG.dt
    rng = IG.rng
    
    #   (force update)
    f = IG.GetForce(q,getcovariance=False )
    
    # Random number
    R = rng.randn( p.shape[0], p.shape[1] )
    
    # Update position
    q += IG.h * f + dt*R
    
    return q , p , extra



def Setup( IG ):

    print ("Using SGLD integrator")
    print (" : Welling and Teh, ICML (2011)")
    print (" : see http://people.ee.duke.edu/~lcarin/398_icmlpaper.pdf")
 

    IG.extra = []
    IG.h = IG.dt * IG.dt * 0.5





