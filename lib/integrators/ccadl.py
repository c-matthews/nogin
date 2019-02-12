import numpy as np

def Step(IG, q,p,extra):
 
    dt = IG.dt
    dt2 = IG.dt2
    rng = IG.rng
    c3 = IG.c3
    x = extra[0]
    
    # Position update
    q += dt * p
    
    #   (force update)
    f, C = IG.GetForce(q)
    
    # Random number
    R = c3 * rng.randn( p.shape[0], p.shape[1] )
    
    # Update momentum
    p += dt * f - (dt*dt2) * IG.m.CovVec( C , p ) - (dt* x) * p
    p += R
    
    # Update x
    x += dt * ( np.sum( p*p ) / p.size - 1.0  )
    
    extra = [x]
    
    return q , p , extra



def Setup( IG ):

    print("Using CC-ADL integrator")
    print(" : Shang, Zhu, Leimkuhler and Storkey, NIPS (2015)")
    print(" : see https://arxiv.org/abs/1510.08692")
 
    x = 0
    IG.extra = [x ]
    IG.c3 = np.sqrt( 2 * IG.dt * IG.g )

def ResetFriction( IG ):
    IG.c3 = np.sqrt( 2 * IG.dt * IG.g )




