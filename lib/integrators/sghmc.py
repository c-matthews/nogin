import numpy as np

def Step(IG, q,p,extra):
 
    
    dt = IG.dt
    dt2 = IG.dt2
    sq2dt = IG.sq2dt
    rng = IG.rng

    # A
    q += dt * p

    #   (force update)
    f, c = IG.GetForce(q)
    
    R =  rng.randn( p.shape[0], p.shape[1] )
    
    if (IG.adaptsghmc):
        success = False
        for ii in xrange(100):
            CminusB = IG.Cmatrix - dt2 * c
            
            try:
                CminusB = np.linalg.cholesky(CminusB)
                success = True
                break
            except:
                IG.g *= 1.5
                IG.Cmatrix  = IG.g*np.eye(IG.m.q.size)
                IG.c1 = IG.dt * IG.g
                print "  >> AdaptSGHMC: Increasing friction to " + str(IG.g)

        if (not success):
            raise Exception('Could not increase friction enough')

    
    else:
        CminusB = IG.Cmatrix - dt2 * c
        CminusB = np.linalg.cholesky(CminusB)

    

    p += dt * f - IG.c1 * p + sq2dt*np.dot(CminusB, R )
    
    
    return q , p , extra



def Setup( IG ):

    print "Using SGHMC integrator"
    print " : Chen, Fox and Guestrin, ICML (2014)"
    print " : see https://arxiv.org/abs/1402.4102"
 

    IG.extra = []
    IG.sq2dt = np.sqrt(2*IG.dt)
    IG.dtsq = 0.5*IG.dt
    IG.Cmatrix  = IG.g*np.eye(IG.m.q.size)
    IG.c1 = IG.dt * IG.g




