from scipy.fftpack import dct,idct
import numpy as np

class DCT_lowpass_filter:
    """
    author:@mtchibozo
    date:10/30/2020
    
    This class applies Discrete Cosine Transform smoothing to an input time series array X.
    Each time series in X is approximated using nb_coefs DCT coeficients. Instead of using a DCT and inverse DCT with nb_coefs, we keep the same dimensionality as X for both.
    
    Parameters
    ----------
    X: array-like of shape (nb_timeseries,context_scale)
    
    
    Attributes
    ----------
    nb_coefs: int
    The number of DCT coefficients which are kept for the DCT transform (all other coefficients are set to zero).
    nb_coefs = 13 was determined after applying the elbow method to the plot of np.linalg.norm(X-X_smoothed) with varying values of nb_coefs.

    X_dct: array-like of shape (nb_timeseries,context_scale)
    Contains the nb_coefs first DCT coefficient for each time time series in X, and zero-pads the remaining (context_scale-nb_coefs) coefficients.
    This ensures we are smoothing the function, whereas using fewer coefficients keeps generates a spiky reconstruction for each time series.
    
    X_smoothed: array-like of shape (nb_timeseries,context_scale)
    Contains the smoothed reconstruction of X using 13 DCT coefficients and zero-padding the rest.
    Smoothing is equivalent to applying a low-pass filter which removes the noise coefficients.
        
    Example
    -------
    >>> lowpass_filter = DCT_lowpass_filter()
    >>> X_dct_reconstructed = lowpass_filter.fit_transform(X_scaled)
    >>> X_dct = lowpass_filter.X_dct

    
    """
    
    def __init__(self,):
        self.nb_coefs = 13
        self.X_dct = None
        self.X_smoothed = None

    def fit(self, X):
        self.X_dct = dct(X,norm="ortho",n=X.shape[1])
        self.X_dct[:,self.nb_coefs:] = np.zeros((X.shape[0],X.shape[1]-self.nb_coefs))
        self.X_smoothed = idct(X_dct,norm="ortho")

    def fit_transform(self, X):
        self.X_dct = dct(X,norm="ortho",n=X.shape[1])
        self.X_dct[:,self.nb_coefs:] = np.zeros((X.shape[0],X.shape[1]-self.nb_coefs))
        self.X_smoothed = idct(self.X_dct,norm="ortho")
        return self.X_smoothed
    
    def transform(self, X):
        return self.X_smoothed