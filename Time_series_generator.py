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