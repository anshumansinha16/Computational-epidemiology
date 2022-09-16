import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

g= nx.read_edgelist('example.txt', create_using = nx.Graph(), nodetype= int)

beta = 0.1
gamma = 0.05
max_time = 200
t = np.linspace(0, max_time, max_time)

S0 = 0.90
I0 = 0.10
R0 = 0
N = 100

import math
I0, R0 = math.floor(N*(0.10798003802281400)), math.floor(N*(0.0020386203150461700))
    # Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0

num_run = 50

count_sz = [[0 for x in range(max_time)] for y in range(num_run)] 
count_iz = [[0 for x in range(max_time)] for y in range(num_run)] 
count_rz = [[0 for x in range(max_time)] for y in range(num_run)] 

count_s= [0 for x in range(max_time)]
count_i= [0 for x in range(max_time)]
count_r= [0 for x in range(max_time)]

import random
a = [1 for x in range(N+1)]
for p in range(I0):
    num = random.randint(0, N)
    a[num]=0

for zi in range(num_run):

  print(zi)
  
  a = [1 for x in range(N+10)]

  for p in range(I0):
    num = random.randint(0, N)
    a[num]=0

  for iter in range(max_time):
    
    cs = 0
    ci = 0
    cr = 0

    for i in a:
      if(i == 1):
        cs = cs+1
      elif(i == 0):
        ci = ci+1
      else:
        cr = cr+1 

    count_s[iter] = cs/N
    count_i[iter] = ci/N
    count_r[iter] = cr/N
    
    for u in g.nodes(): # iterate over all nodes
      #print(u)

      if(a[u]==1):
          nb = nx.all_neighbors(g,u)
          
          for i in nb:
            num = random.randint(0, 99)
            #print('num',num)
            if (i == num): 
              if(a[i]==0):
                r = round(random.uniform(0, 1),5)
                if(r<beta):
                  a[u] = 0        

      elif(a[u]==0):

            r = round(random.uniform(0, 1),5)
            if(r<gamma):
                a[u] = -1      
      
      
      nx.set_node_attributes(g, a[u],'i')

  count_sz[zi][:] = count_s
  count_iz[zi][:] = count_i
  count_rz[zi][:] = count_r

  nms= np.mean(count_sz, axis=0)
nmi= np.mean(count_iz, axis=0)
nmr= np.mean(count_rz, axis=0)

# Plotting both the curves simultaneously

from matplotlib.pyplot import figure

figure(figsize=(8, 6), dpi=80)

plt.plot(t, nms, color='b', label='S_SIR network')
plt.plot(t, nmi, color='r', label='I_SIR network')
plt.plot(t, nmr, color='g', label='R_SIR network')
#plt.plot(t, S, color='b', linestyle='dashed', label='S_SIR ode')
#plt.plot(t, I, color='r', linestyle='dashed', label='I_SIR ode')
#plt.plot(t, R, color='g', linestyle='dashed', label='R_SIR ode')
  
# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Time (days)")
plt.ylabel(" Fraction of S/I/R ")
plt.title("SIR network model on example.txt ")

plt.rc('xtick', labelsize=5) 
plt.rc('ytick', labelsize=5) 
  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
  
# To load the display window
plt.show()