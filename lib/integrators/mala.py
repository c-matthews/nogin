import numpy as np

def Step(IG, q,p,extra):
 
    dt = IG.dt
    rng = IG.rng
    
    oldf, oldeng, acc = extra
    
    
    # Random number
    R = rng.randn( p.shape[0], p.shape[1] )
    
    # Update position
    qq = q +  IG.h * oldf + dt*R
    
    #   (force update)
    ff = IG.GetForce(qq,getcovariance=False )
    neweng = IG.m.llh
    
    if (np.isnan(neweng) or np.isinf(neweng) ):
        IG.m.llh = oldeng
        return q,p,extra
    
    Ratio = (neweng - oldeng)
    
    zz1 = qq - q - IG.h * oldf
    zz2 = q - qq - IG.h * ff
    Ratio += np.sum( zz1*zz1 ) / ( 4.0 * IG.h )
    Ratio -= np.sum( zz2*zz2 ) / ( 4.0 * IG.h )

    lrand = np.log( rng.rand() )

    if (Ratio < lrand):
        IG.m.llh = oldeng
        return q,p,extra

    acc += 1
    IG.acc = acc
    extra2 = [ff,neweng,acc]

    return qq , p , extra2



def Setup( IG ):

    print("Using MALA integrator")
    print(" : Roberts and Rosenthal, RSS-B (1998)")
    print(" : see https://www.jstor.org/stable/pdf/2985986.pdf")
  
    
    ceng = IG.m.llh
    
    acc = 0
     
    IG.extra = [np.copy(IG.m.F),ceng,acc]
    IG.h = IG.dt * IG.dt * 0.5





