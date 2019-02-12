import numpy as np

def Step(IG, q,p,extra):
 
    dt = IG.dt
    dt2 = IG.dt2
    rng = IG.rng
    x = extra[0]
    
    # A
    q += dt2 * p
    
    # D
    x += dt2 * ( np.sum( p*p ) - p.size  )
    
    # O
    R = rng.randn( p.shape[0], p.shape[1] )
    if (x==0):
        p += np.sqrt(dt) * R
    else:
        p = np.exp(-x * dt ) * p + np.sqrt( (1 - np.exp(-2*x*dt))/(2.0*x)) * R
    
    # D
    x += dt2 * ( np.sum( p*p ) - p.size  )

    # A
    q += dt2 * p

    #   (force update)
    f = IG.GetForce(q, getcovariance=False)

    # B
    p += dt * f
    
    extra = [x]
    
    return q , p , extra



def Setup( IG ):

    print "Using SGNHT-S integrator"
    print " : Leimkuhler and Shang, SIAM J. Sci. Comput. (2016)"
    print " : see https://arxiv.org/abs/1505.06889"
 
    x = 0
    IG.extra = [x ]

