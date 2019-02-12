import numpy as np

def LLH( Model, q, idxs ):

    X = Model.X[:,idxs].T
    t = Model.t[:,idxs].T
    alpha = Model.alpha
    
    # Prior
    
    Vprior = -0.5*np.sum(q**2) / alpha
    Fprior = -q / alpha
    
    
    # LLH

    tv = np.dot( X,q)
    exptv = np.exp(-tv)
    
    V = np.sum( tv*t ) - np.sum( np.log( 1+exptv ) ) - np.sum(tv )
    F = X * (t - 1.0/(1.0+exptv))
    
    return V, F.T, Vprior, Fprior

    


def Setup( Model ):

    Model.X = Model.data['X']
    Model.t = Model.data['t']
    Model.alpha = 100.0

    Model.DataSize = Model.X.shape[1]
    
    # Whiten the data
    Model.X = Model.X - np.mean(Model.X,axis=1,keepdims=True)
    Model.X = Model.X / np.std( Model.X, axis=1,keepdims=True)
    
    # Add the constants
    Model.X = np.insert( Model.X , 0 , 1.0 , axis=0 )

    print "Bayesian Linear Regression model with Gaussian prior"

