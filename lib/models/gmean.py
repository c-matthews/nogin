import numpy as np

def LLH( Model, q, idxs ):

    X = Model.X[:,idxs]

    F = X - q

    llh = -np.sum( (X-q)**2 )*0.5

    return llh, F, 0 , 0

    


def Setup( Model ):

    Model.X = Model.data['X']

    Model.DataSize = Model.X.shape[1]
    

    print "Simple Gaussian model with Gaussian distributed means"

