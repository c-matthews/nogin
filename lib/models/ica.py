import numpy as np

def LLH( Model, q, idxs ):

    X = Model.X[:,idxs]
    alpha = Model.alpha
    
    # Prior
    
    Vprior = -0.5*np.sum(q**2) / alpha
    Fprior = -q / alpha
    
    
    # LLH
    
    D,N = X.shape
    
    
    W = np.reshape( q , (D,D) ).T
    Winv = np.linalg.inv(W)
    Wdet = np.linalg.det(W)
    
    WX2 = -0.5 * np.dot( W, X )
    
    em2x = np.exp(2*WX2 )
    tanhx = (1.0 - em2x) / (1.0 + em2x )
    logcoshx = np.log1p( em2x ) - WX2 - 0.69314718
    
    V = N * np.log( np.abs( Wdet ) ) - 2*np.sum( logcoshx )
    
    F = -np.einsum('ij,kj->kij',tanhx,X).reshape( (D*D,-1 ) )
    F = F + np.reshape( Winv.flatten(), q.shape )
    
    return V, F, Vprior, Fprior

    


def Setup( Model ):

    Model.X = Model.data['X']

    Model.DataSize = Model.X.shape[1]

    Model.alpha = 1.0

    print ("Independent Component Analysis model with Gaussian prior")

