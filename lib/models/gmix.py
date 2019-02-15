import numpy as np

def LLH( Model, q, idxs ):

    X = Model.X[:,idxs]

    w = np.array([0.5,1]).reshape([2,1])

    dmu = X - q
    demu = np.exp(-dmu**2/2) * w 
    
    mix = np.sum( demu,0)

    llh = np.sum( np.log(mix) )
    
    F = dmu * demu / mix

    return llh, F, 0 , 0

    


def Setup( Model ):

    Model.X = Model.data['X']

    Model.DataSize = Model.X.shape[1]
    
    print ("Inferring centres for a two-Gaussian mixture model")

