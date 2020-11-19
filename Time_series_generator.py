import numpy as np
import numpy.random as rd
import pandas as pd

class TS_generator:
  def __init__(self, nb_timeseries=2000, chunk_size=100):
    
    self.chunk_size = chunk_size
    self.nb_timeseries = nb_timeseries

    #Retrieve the stocks names
    self.symbols = pd.read_csv('https://raw.githubusercontent.com/Amelrich/Capstone-Fall-2020/master/sp500.csv', index_col=False)
    self.symbols = list(self.symbols['Symbol'].values)
    self.symbols = sorted(self.symbols)
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
    #Return a list of time series dataframes
    return self.list_df

  def get_array(self):
    #Return adjusted close array
    close_array = np.zeros((self.nb_timeseries, self.chunk_size))

    for i in range(self.nb_timeseries):
      close_array[i,:] = self.list_df[i]['Adj Close'].to_numpy()

    return close_array

from scipy.fftpack import idct

class Synthetic_TS_generator:
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
        self.time_series = None
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
        
        