import numpy as np
import numpy.random as rd
import pandas as pd
import os

def scale_min_max(ts):
    return (ts - ts.min(axis=1).reshape(-1, 1))/ts.ptp(axis=1).reshape(-1, 1)

def weight_ts(ts, reverse=False, alpha=1.2):
    if reverse: # weights increase as you go back in time
        weights = np.array([np.sqrt(alpha**(ts.shape[1] - i)) for i in range(ts.shape[1])])
    else:
        weights = np.array([np.sqrt(alpha**(i+1)) for i in range(ts.shape[1])])
    result = np.multiply(ts, weights)
    return np.square(weights), result

class TS_generator:

    def __init__(self, nb_timeseries=2000, chunk_size=100):
    
        self.chunk_size = chunk_size
        self.nb_timeseries = nb_timeseries

        #Retrieve the stocks names
        self.symbols = pd.read_csv('sp500.csv', index_col=False)
        self.symbols = list(self.symbols['Symbol'].values)
        self.symbols = ['BF' if x=='BF.B' else x for x in self.symbols]
        self.symbols = ['BKR' if x=='BKR.B' else x for x in self.symbols]
        self.list_df = []
        #Build the random time series
        self.build_()

    def build_(self):    
        self.files = os.listdir('data')
        for _ in range(self.nb_timeseries):
            #Pick a random stock
            stock_file = self.symbols[rd.randint(len(self.symbols))] + '.csv'
            if stock_file in self.files:
                TS = pd.read_csv('data/' + stock_file)  
                #Pick a random starting point
                timemax = len(TS) - self.chunk_size
                if timemax > 0:
                    start = rd.randint(timemax)
                    stock_df = TS[start : start+self.chunk_size]
                    self.list_df.append(stock_df)

    def get_list_of_df(self):
        return self.list_df

    def get_array(self):
        #Return adjusted close array
        n_stock_files = len(self.list_df)
        close_array = np.zeros((n_stock_files, self.chunk_size))
        for i in range(n_stock_files):
            close_array[i,:] = self.list_df[i]['Adj Close'].to_numpy()
        return close_array