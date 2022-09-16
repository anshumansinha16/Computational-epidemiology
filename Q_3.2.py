import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

g= nx.read_edgelist('network.txt', create_using = nx.Graph(), nodetype= int)

# import dataset

import pandas as pd
import numpy as np
from pandas import Series, DataFrame

filename = '/content/Ratings.timed.csv'
df = pd.read_csv(filename)
df.shape

df["date"] = pd.to_datetime(df["date"])



G = nx.Graph()
for u in g.nodes():
  #u = 5525
  nb = nx.all_neighbors(g,u) # neighbours of node u
  p = df.loc[df['userid'] == u] # extract all entries (all columns) with  node u
  r = len(p) # Av (Total number of tries u does to infect others)


  if (r >0): # If the node has some enteries to infect others
            
                for i in nb: # loop over all the neibours of user u
                    count = 0 # counter for succesfull tries
                    t = df.loc[df['userid'] == i]
                    if(len(t)>0):

                      for mid,did in zip(p.movieid, p.date): # extract the movie ID column for later comparison
              
                        for mid_1,did_1 in zip(t.movieid, t.date):

                          if( mid == mid_1) and (did<=did_1):# && 
                             if (did==did_1):
                               if(df[df['movieid'] == mid].index[0] > df[df['movieid'] == mid_1].index[0] ): # tie breaker
                                count = count+1
                             else:
                               count = count+1

                      if(count>0):
                        #print('weight=', count/r)  
                        G.add_edge(u, i, weight= count/r)