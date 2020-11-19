import numpy as np
import numpy.random as rd
import pandas as pd

from sklearn.base import ClusterMixin, BaseEstimator, TransformerMixin

from scipy.fftpack import dct, idct


class TS_generator:
    """
    author:@Amelrich
    date:11/05/2020
    
    It generates randomly picked timeseries among all the available timeseries presents on the market period.
    Currently you can access to the generated time series with either dataframes or arrays.

    Parameters
    ----------
    nb_timeseries: int, default=2000
        The number of random timeseries the generator should pick.

    chunk_size: int, default=60
        The time window in days of timeseries generated.
    
    Example
    -------
    >>> gen = TS_generator()
    >>> X = gen.get_array(X)


    Methods
    -------
    get_list_of_df() - Returns a list of timeseries contained in a dataframe.
    Returns:
        l_df: list of length (nb_timeseries,) with dataframes of length (chunk_size,)
        List of timeseries contained in a dataframe.


    get_array() - Returns the random timeseries as an array.
    Returns:
        X: ndarray of shape (nb_timeseries, chunk_size)
        Array of timeseries.

    """


    def __init__(self, nb_timeseries:int=2000, chunk_size:int=100):

        self.chunk_size = chunk_size
        self.nb_timeseries = nb_timeseries

        #Retrieve the stocks names
        self.symbols = pd.read_csv('https://raw.githubusercontent.com/Amelrich/Capstone-Fall-2020/master/sp500.csv', index_col=False)
        self.symbols = list(self.symbols['Symbol'].values)
        self.symbols = ['BF-B' if x=='BF.B' else x for x in self.symbols]
        self.symbols = ['BRK-B' if x=='BRK.B' else x for x in self.symbols]

        self.list_df = []

        #Build the random time series
        self.build_()

    def build_(self):

        TS_list = []
        indexes = [] #Starting date indexes
        total_len = 0

        for stock in self.symbols:
            TS = pd.read_csv('data/'+stock+'.csv')
            TS_list.append(TS)
            indexes += list(range(total_len, total_len + len(TS) - self.chunk_size))
            total_len += len(TS)

        TS = pd.concat(TS_list, ignore_index=True)
        del(TS_list)

        #Pick random starting dates
        random_starts = rd.choice(indexes, self.nb_timeseries)

        for start in random_starts:
            self.list_df.append( TS[start : start+self.chunk_size] )

        del(TS)

    def get_list_of_df(self):
        return self.list_df

    def get_array(self):
        #Return adjusted close array
        close_array = np.zeros((self.nb_timeseries, self.chunk_size))

        for i in range(self.nb_timeseries):
            close_array[i,:] = self.list_df[i]['Adj Close'].to_numpy()

        return close_array



class MedianScaler(TransformerMixin, BaseEstimator):
    """
    author:@Amelrich
    date:11/05/2020
    
    This class applies a median scaling. 
    It is similar to a standard scaler by replacing the mean by the median and the standard deviation by MAD.
    
    Attributes
    ----------
    median: ndarray of shape (n_features,)
        Per feature median.

    mad: ndarray of shape (n_features,)
        Per feature MAD.
    
    Example
    -------
    >>> med_scaler = MedianScaler()
    >>> X_train = med_scaler.fit_transform(X)


    Methods
    -------
    fit(X, y=None) - Compute the medians and MAD
    Parameters:
        X: ndarray of shape (n_samples, n_features)
        Training instances to cluster.

        y: Ignored
        Not used, present here for API consistency by convention.

    Returns:
        self
        Fitted estimator.


    transform(X) - Transform the data using the fitted medians and MAD.
    Parameters:
        X: ndarray of shape (n_samples, n_features)
        Data to transform.

    Returns:
        Xt: ndarray of shape (n_samples, n_features)
        Transformed data.


    fit_transform(X, y=None) - Fit the medians and MAD before returning the transformed data. 
    Equivalent to MedianScaler().fit(X).transform(X) but done more efficiently.
    Parameters:
        X: ndarray of shape (n_samples, n_features)
        Data to transform.

        y: Ignored
        Not used, present here for API consistency by convention.

    Returns:
        Xt: ndarray of shape (n_samples, n_features)
        Transformed data.
    """

    def __init__(self, **kwargs):
        super(MedianScaler, self).__init__(**kwargs)
        self.median = None
        self.mad = None

    def fit(self, X):
        self.median = np.median(X, axis=0)
        self.mad = np.median( np.abs(X - self.median), axis=0 )
        return self

    def fit_transform(self, X):
        self.median = np.median(X, axis=0)
        self.mad = np.median( np.abs(X - self.median), axis=0 )
        return (X - self.median) / self.mad

    def transform(self, X):
        return (X - self.median) / self.mad





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
        self.X_smoothed = idct(self.X_dct,norm="ortho")

    def fit_transform(self, X):
        self.X_dct = dct(X,norm="ortho",n=X.shape[1])
        self.X_dct[:,self.nb_coefs:] = np.zeros((X.shape[0],X.shape[1]-self.nb_coefs))
        self.X_smoothed = idct(self.X_dct,norm="ortho")
        return self.X_smoothed
    
    def transform(self, X):
        return self.X_smoothed



from scipy.fftpack import idct

class Synthetic_TS_generator:
    """
    author:@mtchibozo
    date:11/19/2020
    
    
    This class generates random time series with multiscale patterns.
    To ensure multiscale properties, we create one low frequency and one high frequency DCT coefficient while mainting all other coefficients to zero. We also add an attribute to add noise to the smooth signal.
    Possible impovement: as is, the code only uses 60 coefficients (or context_scale). We could introduce more coefficients to generate a more complex time series, then subsample a time series of length context_scale.
    
    
    Parameters
    ----------
   
    
    Attributes
    ----------
    nb_coefs: int
    The number of DCT coefficients which are kept for the DCT transform (all other coefficients are set to zero).
    nb_coefs = 13 was determined after applying the elbow method to the plot of np.linalg.norm(X-X_smoothed) with varying values of nb_coefs.
    
     nb_timeseries: int (default 3000) 
     Number of multiscale synthetic time series we want to generate.
     
     chunk_size: int (default 60)
     Length of each synthetic time series.
     
     long_scale: Boolean (default True)
     Whether or not we want a long scale trend in the signal (=low frequency).
     
     short_scale: Boolean (default True)
     Whether or not we want a short scale trend in the signal (=high frequency)
     
     noise: Boolean (default False)
     Whether or not we want noise in the signal (=very high frequency).
     
     low_freq_range: tuple (default (1,min(4,chunk_size))
     Range of DCT coefficients which we consider for the long scale.
     
     high_freq_range: tuple (default (min(7,chunk_size),min(10,chunk_size)))
     Range of DCT coefficients which we consider for the short scale.
     
     noise_freq_range: tuple (default (min(15,chunk_size),chunk_size))
     Range of DCT coefficients which we consider for noise
     
     dct_coefs: np.array (default np.zeros((nb_timeseries,chunk_size)))
     Dataset containing the DCT coefficients associated to the synthetic time series.
     
     time_series: np.array (default np.zeros((nb_timeseries,chunk_size)))
     Dataset containing the synthetic time series. The time series are Min-Max scaled.
     time series = inverse DCT (DCT coefficients).

        Example
    -------
    >>> stg = Synthetic_TS_generator(noise=True)
    >>> X_synthetic = stg.get_array()

    
    """

    def __init__(self,nb_timeseries=3000,chunk_size=60,long_scale=True,short_scale=True,noise=False):
        self.nb_timeseries = nb_timeseries
        self.chunk_size = chunk_size
        self.long_scale = long_scale
        self.short_scale = short_scale
        self.noise = noise
        self.low_freq_range = (1,min(4,chunk_size))
        self.high_freq_range = (min(7,chunk_size),min(10,chunk_size))
        self.noise_freq_range = (min(15,chunk_size),chunk_size)
        self.dct_coefs = np.zeros((nb_timeseries,chunk_size))
        self.time_series = np.zeros((nb_timeseries,chunk_size))
        #Build the random time series
        self.build_()
        
    def scale(self, matrix):
        norm_matrix = matrix.copy()
        for row in range(matrix.shape[0]):
            norm_matrix[row,:] = (matrix[row,:]-np.min(matrix[row,:]))/(np.max(matrix[row,:])-np.min(matrix[row,:]))
        return norm_matrix

        
    def build_(self):

        #Build long scale
        long_scale_coef_limit_0 = self.low_freq_range[0]
        long_scale_coef_limit_1 = self.low_freq_range[1]
        long_scale_coefs_idx = np.random.multinomial(1, [1/(long_scale_coef_limit_1-long_scale_coef_limit_0)]*(long_scale_coef_limit_1-long_scale_coef_limit_0),size=self.nb_timeseries) #pick a long scale coefficient at random
        long_scale_coefs_vals = np.random.uniform(low=-2,high=2,size=self.nb_timeseries)
        long_scale_coefs = np.multiply(long_scale_coefs_vals.reshape(-1,1),long_scale_coefs_idx)
        if self.long_scale == True:
            self.dct_coefs[:,long_scale_coef_limit_0:long_scale_coef_limit_1] = long_scale_coefs

        #Build short scale
        short_scale_coef_limit_0 = self.high_freq_range[0]
        short_scale_coef_limit_1 = self.high_freq_range[1]
        short_scale_coefs_idx = np.random.multinomial(1, [1/(short_scale_coef_limit_1-short_scale_coef_limit_0)]*(short_scale_coef_limit_1-short_scale_coef_limit_0),size=self.nb_timeseries) #pick a long scale coefficient at random
        short_scale_coefs_vals = np.multiply(long_scale_coefs_vals/2,np.random.binomial(n=1,p=0.5,size=self.nb_timeseries)*2-np.ones(self.nb_timeseries))
        short_scale_coefs = np.multiply(short_scale_coefs_vals.reshape(-1,1),short_scale_coefs_idx)
        if self.short_scale == True:
            self.dct_coefs[:,short_scale_coef_limit_0:short_scale_coef_limit_1] = short_scale_coefs

        #Build noise
        noise_scale_coef_limit_0 = self.noise_freq_range[0]
        noise_scale_coef_limit_1 = self.noise_freq_range[1]
        noise_scale_coefs_idx = np.random.multinomial(3, [1/(noise_scale_coef_limit_1-noise_scale_coef_limit_0)]*(noise_scale_coef_limit_1-noise_scale_coef_limit_0),size=self.nb_timeseries) #pick a long scale coefficient at random
        noise_scale_coefs_vals = np.multiply(long_scale_coefs_vals/6,np.random.binomial(n=1,p=0.5,size=self.nb_timeseries)*2-np.ones(self.nb_timeseries))
        noise_scale_coefs = np.multiply(noise_scale_coefs_vals.reshape(-1,1),noise_scale_coefs_idx)
        if self.noise == True:
            self.dct_coefs[:,noise_scale_coef_limit_0:noise_scale_coef_limit_1] = noise_scale_coefs


    def get_array(self):
        self.time_series = self.scale(idct(self.dct_coefs))
        return self.time_series
        


class KMedians(ClusterMixin, BaseEstimator):
    """
    author:@Amelrich
    date:11/05/2020
    
    This class applies the KMedians algorithm which is the equivalent of KMeans for L1 distance.
    
    Parameters
    ----------
    n_clusters: int, default=5
        The number of clusters the algorithm aim to find.

    max_iter: int, default=100
        The maximum number of iteration before stopping the fitting process.

    tol: float, default=0.001
        The value below which a change of inertia stops the fitting process.
    
    
    Attributes
    ----------
    clusters_centers_: ndarray of shape (n_cluster, n_features)
        Coordinates of cluster centers.

    labels_: ndarray of shape (n_samples,)
        Labels of each point

    inertia_: float
        Sum of L1 distances of samples to their closest cluster center.
        
    Example
    -------
    >>> kmedians = KMedians()
    >>> labels = kmedians.fit_predict(X_train)


    Methods
    -------
    fit(X, y=None) - Compute the kmedians clustering
    Parameters:
        X: ndarray of shape (n_samples, n_features)
        Training instances to cluster.

        y: Ignored
        Not used, present here for API consistency by convention.

    Returns:
        self
        Fitted estimator.


    predict(X) - Predict the closest cluster each sample in X belongs to.
    Parameters:
        X: ndarray of shape (n_samples, n_features)
        New data to predict.

    Returns:
        labels
        Index of the cluster each sample belongs to.


    fit_predict(X, y=None) - Compute the kmedians clustering and returns labels. 
    Equivalent to kmedians.fit(X).predict(X) but done more efficiently.
    Parameters:
        X: ndarray of shape (n_samples, n_features)
        Training instances to cluster.

        y: Ignored
        Not used, present here for API consistency by convention.

    Returns:
        labels
        Index of the cluster each sample belongs to.
    """


    def __init__(self, n_clusters:int=5, max_iter:int=100, tol:float=0.001, **kwargs):
        super(KMedians, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol


    def fit(self, X, y=None):
        self._init_centers(X)
        self._update_dist_labels(X)

        inertia = np.inf
        self._update_inertia()

        iter = 0
        while (iter < self.max_iter) and ((inertia - self.inertia_) > self.tol):
            self._update_centers(X)
            self._update_dist_labels(X)
            inertia = self.inertia_
            self._update_inertia()
            iter += 1
        
        return self


    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_


    def predict(self, X):
        dist_predict = np.empty((X.shape[0],self.n_clusters))
        for i in range(X.shape[0]):
            dist_predict[i,:] = np.sum( np.abs(X[i,:] - self.cluster_centers_), axis=1 )

        return np.argmin(self.dist, axis=1)

    
    def _init_centers(self, X):
        idx = []
        idx.append(rd.choice(X.shape[0]))

        for k in range(1, self.n_clusters+1):
            self.cluster_centers_ = X[idx, :]
            self.dist = np.empty((X.shape[0], k))

            for i in range(X.shape[0]):
                self.dist[i,:] = np.sum( np.abs(X[i,:] - self.cluster_centers_), axis=1 )

            idx.append( np.argmax(self.dist.min(axis=1)) )

        del(idx)


    def _update_dist_labels(self, X):
        for i in range(X.shape[0]):
            self.dist[i,:] = np.sum( np.abs(X[i,:] - self.cluster_centers_), axis=1 )

        self.labels_ = np.argmin(self.dist, axis=1)


    def _update_centers(self, X):
        for k in range(self.n_clusters):
            self.cluster_centers_[k,:] = np.median(X[self.labels_==k,:], axis=0)
            

    def _update_inertia(self):
        self.inertia_ = 0
        for k in range(self.n_clusters):
            self.inertia_ += np.sum(self.dist[self.labels_ == k,k])