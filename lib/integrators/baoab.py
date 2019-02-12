import numpy as np

def Step(IG, q,p,extra):

    f = extra[0]
    
    dt2 = IG.dt2
    c1 = IG.c1
    c3 = IG.c3
    rng = IG.rng

    # B
    p += dt2 * f

    # A
    q += dt2 * p

    # O
    p = c1*p + c3*rng.randn( p.shape[0], p.shape[1] )

    # A
    q += dt2 * p
    
    #   (force update)
    f = IG.GetForce(q,getcovariance=False)

    # B
    p += dt2 * f

    extra = [f]
    return q , p , extra



def Setup( IG ):

    print("Using BAOAB integrator")
    print(" : Leimkuhler and Matthews, AMRX (2013)")
    print(" : see https://arxiv.org/abs/1203.5428")
    if IG.nonoise:
        print(" : (Running as optimizer)")


    IG.extra = [np.copy(IG.m.F)]
    ResetFriction( IG )

def ResetFriction( IG ):
    IG.c1 = np.exp(- IG.dt * IG.g )
    if IG.nonoise:
        IG.c3 = 0
    else:
        IG.c3 = np.sqrt(1.0 - IG.c1**2 )



