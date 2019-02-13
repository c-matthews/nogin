import numpy as np
from scipy.sparse.linalg import cg

def Step(IG, q,p,extra):
 
    
    dt2 = IG.dt2
    dt24 = IG.dt24
    c1 = IG.c1
    c3 = IG.c3
    rng = IG.rng

    # A
    q += dt2 * p
    
    #   (force update)
    f, c = IG.GetForce(q)
    
    # B
    p += dt2 * f
    
    # O
    R = c3*rng.randn( p.shape[0], p.shape[1] )
    p += R
    
    p = IG.NOGINsolve(c1,dt24,c,p)
    #p = np.dot( (np.eye(p.size)*(1-c1) - dt24*c ), p )
    p = (1-c1)*p - dt24*IG.m.CovVec(c,p)
    p += R
    
    # B
    p += dt2*f
    
    # A
    q += dt2*p
 
    return q , p , extra



def Setup( IG ):

    print("Using NOGIN integrator")
    print(" : Matthews and Weare (2018)")
    print(" : see https://arxiv.org/abs/1805.08863")

    IG.extra = [np.copy(IG.m.F)]
    cc = np.exp(- IG.g * IG.dt )
    IG.c1 = ( 1 - cc ) / (1.0+cc)
    IG.c3 = np.sqrt( IG.c1 )
    IG.dt24 = 0.25* IG.dt**2

    IG.NOGINsolve = ApproxSolve
     
    if (IG.noginexact):
        IG.NOGINsolve = ExactSolve
        print(" : (using exact solve)")
    else:
        if (IG.nogincg):
            IG.NOGINsolve = CGSolve
            print(" : (using conjugate gradient)")


def CGSolve(c1,dt24,c,p):
    tv = cg(np.eye(p.size)*(1+c1) + dt24*c , p, x0=(p/(1.0+c1)) )[0]
    return tv[:,np.newaxis]

def ExactSolve(c1,dt24,c,p):

    return np.linalg.solve( (np.eye(p.size)*(1+c1) + dt24*c ), p )

def ApproxSolve(c1,dt24,c,p):

    # A positive definite approximation to the exact solve, given above

    cp = np.dot( c , p )
    ccp = np.dot( c , cp )
    fac = dt24 / (1.0 + c1 )

    return (p - fac*cp + (fac*fac)*ccp)/(1.0+c1)

def covhistsolve(c1,dt24,c,p):
    # see https://en.wikipedia.org/wiki/Woodbury_matrix_identity

    const = dt24 /((c.shape[0]-1)*(1.0+c1 ))
    p1 = np.dot( c , p )
    p2 = cg( np.eye(p1.size) + (const)*np.dot(c,c.T) , p1, x0=p1 )[0][:,np.newaxis]
    p3 = np.dot( c.T , p2 )* (const)

    return (p - p3  ) / (1.0+c1)



def ResetFriction( IG ):

    cc = np.exp(- IG.g * IG.dt )
    IG.c1 = ( 1 - cc ) / (1.0+cc)
    IG.c3 = np.sqrt( IG.c1 )

    
    if (IG.m.CC.size==1):
        evals = IG.m.CC
    else:
        try:
            #if (IG.m.covhist_do):
            #    IG.m.CC = IG.m.covhist - np.mean( IG.m.covhist,axis=0)
            #    IG.m.CC = np.dot( IG.m.CC.T , IG.m.CC ) / (IG.m.CC.shape[0]-1.0)
            evals,evecs = np.linalg.eig( IG.m.CC )
        except:
            print("  >> Failed to optimize friction!")
            return
    
    topev = np.min( evals.real )
    
    IG.c1 -= IG.dt24*topev
    
    smallc1 = np.exp(-1e-4*IG.dt)
    cc2 = ( 1- smallc1 ) / (1.0+smallc1 )
    if (IG.c1 < cc2 ):
        IG.c1 = cc2
    
    IG.c3 = np.sqrt( IG.c1 )

    print( "  >> Effective friction: " + str( -np.log( (1-IG.c1)/(1.0+IG.c1))/IG.dt  ))

    return





